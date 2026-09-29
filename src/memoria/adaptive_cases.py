"""Designed cases and measured studies for claim extraction and entity identity.

Designed cases fix what a conservative reader must do (extract, abstain with a reason, refuse to
merge) on hand-written texts; they are checks of the rules, not evidence about real text. The
studies then measure the same rules on the generated worlds against gold labels: extraction
precision and recall, *uncertain text converted to fact* (must be 0 for the conservative
rules), unresolved rate by reason, claim provenance completeness, and per identity policy the
false merges (a mention resolved to an entity other than its gold entity), ambiguity and misses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from memoria.adaptive_worlds import World, registry_for
from memoria.claims import REASONS, Claim, extract, verify_claim
from memoria.core import Record
from memoria.identity import (
    CONSERVATIVE,
    EXACT_ONLY,
    SIMILARITY,
    IdentityAssertion,
    IdentityPolicy,
    MergeDecision,
    false_merges,
    make_registry,
    merge_decisions,
    resolve_mention,
)
from memoria.statistics import Proportion

CONFIDENCE = 0.95
Row = tuple[str, str, str | None, str | None, str | None, str | None]
# (status, reason, entity, attribute, value, kind)


class ExtractionCase(Record):
    name: str
    text: str
    expected: tuple[Row, ...]


def _x(name: str, text: str, *rows: Row) -> ExtractionCase:
    return ExtractionCase(name=name, text=text, expected=rows)


def _ok(e: str, a: str, v: str, kind: str = "assert", reason: str = "rule_match") -> Row:
    return ("extracted", reason, e, a, v, kind)


def _un(reason: str) -> Row:
    return ("unresolved", reason, None, None, None, None)


EXTRACTION_CASES: tuple[ExtractionCase, ...] = (
    _x("lives-in", "Ana lives in Berlin.", _ok("ana", "home", "berlin")),
    _x("moved-to", "Ana moved to Berlin.", _ok("ana", "home", "berlin")),
    _x("possessive-home", "Ana's home is Rome.", _ok("ana", "home", "rome")),
    _x("works-at", "Ben works at Acme.", _ok("ben", "employer", "acme")),
    _x("joined", "Ben joined Globex.", _ok("ben", "employer", "globex")),
    _x("no-longer-lives", "Ana no longer lives in Paris.", _ok("ana", "home", "paris", "negate")),
    _x("left-employer", "Chen left Initech.", _ok("chen", "employer", "initech", "negate")),
    _x("does-not-live", "Ana doesn't live in Rome.", _ok("ana", "home", "rome", "negate")),
    _x("hedged-think", "I think Ana lives in Rome.", _un("hedged")),
    _x("hedged-maybe-question", "Maybe Ben works at Acme?", _un("hedged")),
    _x("hedged-heard", "Ana works at Acme, I heard.", _un("hedged")),
    _x("bare-question", "Ana lives in Rome?", _un("hedged")),
    _x("pronoun", "She moved to Rome.", _un("pronoun_subject")),
    _x("ambiguous-alias", "Ann lives in Rome.", _un("ambiguous_entity")),
    _x("alias", "Annie lives in Rome.", _ok("ana", "home", "rome")),
    _x("near-collision-exact", "Anna lives in Rome.", _ok("anna", "home", "rome")),
    _x("typo-is-candidate", "Aana lives in Rome.", _un("candidate_entity")),
    _x("unknown-person", "Zed lives in Rome.", _un("unknown_entity")),
    _x(
        "two-facts-one-sentence",
        "Ana lives in Rome and Ben lives in Oslo.",
        _un("multiple_candidates"),
        _un("multiple_candidates"),
    ),
    _x(
        "two-sentences",
        "Ana lives in Rome. Ben works at Acme.",
        _ok("ana", "home", "rome"),
        _ok("ben", "employer", "acme"),
    ),
    _x(
        "pronoun-second-sentence",
        "Ana lives in Rome. She works at Acme.",
        _ok("ana", "home", "rome"),
        _un("pronoun_subject"),
    ),
    _x("embedded-clause", "Ana, who lives in Rome, works at Acme.", _un("no_pattern")),
    _x("trigger-without-pattern", "Ana moved recently to Rome.", _un("no_pattern")),
    _x("no-claim", "The weather is nice."),
    _x("statement", "set ana.home = rome", _ok("ana", "home", "rome", reason="statement")),
    _x(
        "statement-correct",
        "correct ben.employer = acme",
        _ok("ben", "employer", "acme", "correct", "statement"),
    ),
    _x("statement-unknown-entity", "set zed.home = rome", _un("unknown_entity")),
    _x("statement-unknown-attribute", "set ana.pet = rex", _un("unknown_attribute")),
)


def run_extraction_cases() -> tuple[tuple[str, bool, tuple[Row, ...]], ...]:
    """(case, passed, what was produced) on the registry with the near-collision entities."""
    reg = registry_for(True)
    out = []
    for c in EXTRACTION_CASES:
        got = tuple(
            (cl.status, cl.reason, cl.entity, cl.attribute, cl.value, cl.kind)
            for cl in extract(c.text, f"case:{c.name}", reg)
        )
        out.append((c.name, got == c.expected, got))
    return tuple(out)


class IdentityCase(Record):
    name: str
    mention: str
    true_entity: str | None  # None: a person the registry does not know
    expected: dict[str, str]  # policy name -> "outcome:entity" (entity "-" when none)


def _i(name: str, mention: str, true: str | None, c: str, s: str, e: str) -> IdentityCase:
    return IdentityCase(
        name=name,
        mention=mention,
        true_entity=true,
        expected={"conservative": c, "exact-only": e, "similarity": s},
    )


IDENTITY_CASES: tuple[IdentityCase, ...] = (
    _i("exact", "Ana", "ana", "resolved:ana", "resolved:ana", "resolved:ana"),
    _i("alias", "Annie", "ana", "resolved:ana", "resolved:ana", "resolved:ana"),
    _i("shared-alias", "Ann", None, "ambiguous:-", "ambiguous:-", "ambiguous:-"),
    _i("near-collision-exact", "Anna", "anna", "resolved:anna", "resolved:anna", "resolved:anna"),
    _i("typo", "Aana", "ana", "candidates:-", "resolved:ana", "unknown:-"),
    _i("typo-of-near-collision", "Annaa", "anna", "candidates:-", "resolved:anna", "unknown:-"),
    _i("unregistered-near-name", "Benno", None, "candidates:-", "resolved:benn", "unknown:-"),
    _i("unknown", "Zed", None, "unknown:-", "unknown:-", "unknown:-"),
)


def run_identity_cases() -> tuple[tuple[str, str, bool, str, bool], ...]:
    """(case, policy, passed, got, false merge) for every policy."""
    reg = registry_for(True)
    out = []
    for c in IDENTITY_CASES:
        for p in (CONSERVATIVE, EXACT_ONLY, SIMILARITY):
            r = resolve_mention(c.mention, reg, p)
            got = f"{r.outcome}:{r.entity or '-'}"
            wrong = r.outcome == "resolved" and r.entity != c.true_entity
            out.append((c.name, p.name, got == c.expected[p.name], got, wrong))
    return tuple(out)


def merge_cases() -> tuple[tuple[str, MergeDecision, bool], ...]:
    """Identity assertions and their decisions; the flag says whether a merge was false (an
    assertion joining entities the designed truth keeps apart)."""
    reg = make_registry(
        {"ana": ("Ana", []), "anna": ("Anna", []), "ana2": ("Ana Kumar", []), "ben": ("Ben", [])},
        distinct=[("ana", "anna")],
    )
    truth = {"ana": "P1", "ana2": "P1", "anna": "P2", "ben": "P3"}

    def a(x: str, y: str, src: str, ev: str) -> IdentityAssertion:
        return IdentityAssertion(a=x, b=y, source=src, evidence=ev)

    policy = IdentityPolicy(name="min-2-roots", min_roots=2)
    cases = {
        "two-independent-sources-same-person": [
            a("ana", "ana2", "s1", "e1"),
            a("ana", "ana2", "s2", "e2"),
        ],
        "one-source-only": [a("ana", "ana2", "s1", "e1")],
        "declared-distinct-conflict": [a("ana", "anna", "s1", "e1"), a("ana", "anna", "s2", "e2")],
        "undeclared-false-merge": [a("ana", "ben", "s1", "e1"), a("ana", "ben", "s2", "e2")],
    }
    out = []
    for name, xs in cases.items():
        (d,) = merge_decisions(reg, xs, policy)
        out.append((name, d, false_merges([d], truth)[0] > 0))
    return tuple(out)


# --- measured studies on the generated worlds ---------------------------------------------


class ExtractionStats(Record):
    world: str
    policy: str
    reports: int
    certain: int  # reports a careful reader could extract
    uncertain: int
    exact: int  # extracted claim equal to gold (entity, attribute, value, kind)
    wrong_fact: (
        int  # extracted claim that differs from gold, or extracted from a report with no gold fact
    )
    uncertain_as_fact: int  # extracted claims from uncertain reports (silent conversion)
    missed: int  # certain reports with no extracted claim
    unresolved_reports: int
    reasons: tuple[tuple[str, int], ...]
    precision: Proportion | None
    recall: Proportion | None
    conversion: Proportion | None
    claims_verified: Proportion  # extracted claims whose lineage and spans reproduce the text


class IdentityStats(Record):
    world: str
    policy: str
    mentions: int
    correct: int
    false_merges: int  # resolved to an entity other than the gold entity
    ambiguous: int
    candidates: int
    unknown: int
    false_merge_rate: Proportion  # over mentions resolved
    resolved_rate: Proportion  # over mentions


def extraction_stats(
    world: World, claims: dict[str, tuple[Claim, ...]], policy: str
) -> ExtractionStats:
    reasons: Counter[str] = Counter()
    exact = wrong = conv = missed = unres = ok = total = 0
    certain = uncertain = 0
    for r in world.reports:
        cs = claims[r.id]
        got = [c for c in cs if c.status == "extracted"]
        for c in cs:
            if c.status == "unresolved":
                reasons[c.reason] += 1
        unres += any(c.status == "unresolved" for c in cs)
        for c in got:
            total += 1
            ok += not verify_claim(c, r.text)
        if r.certain:
            certain += 1
            hit = [
                c
                for c in got
                if (c.entity, c.attribute, c.value, c.kind)
                == (r.entity, r.attribute, r.value, r.kind)
            ]
            exact += bool(hit)
            wrong += len(got) - len(hit)
            missed += not got
        else:
            uncertain += 1
            conv += len(got)
            wrong += len(got)
    return ExtractionStats(
        world=world.name,
        policy=policy,
        reports=len(world.reports),
        certain=certain,
        uncertain=uncertain,
        exact=exact,
        wrong_fact=wrong,
        uncertain_as_fact=conv,
        missed=missed,
        unresolved_reports=unres,
        reasons=tuple(sorted(reasons.items())),
        precision=Proportion.of(exact, exact + wrong, CONFIDENCE) if exact + wrong else None,
        recall=Proportion.of(exact, certain, CONFIDENCE) if certain else None,
        conversion=Proportion.of(conv, uncertain, CONFIDENCE) if uncertain else None,
        claims_verified=Proportion.of(ok, total, CONFIDENCE),
    )


def identity_stats(world: World, claims: dict[str, tuple[Claim, ...]]) -> tuple[IdentityStats, ...]:
    """The three identity policies on the same mentions: every free-text subject the extractor
    proposed (pronouns are not names). Gold entity is the report's."""
    gold = {r.id: r.entity for r in world.reports}
    ments = [
        (c.proposed_subject, gold[c.experience])
        for cs in claims.values()
        for c in cs
        if c.proposed_subject
        and c.rule != "statement"
        and c.proposed_subject.split()[0].casefold() not in {"she", "he", "they", "it"}
    ]
    out = []
    for p in (CONSERVATIVE, EXACT_ONLY, SIMILARITY):
        n = fm = ok = amb = cand = unk = 0
        for m, g in ments:
            r = resolve_mention(m, world.registry, p)
            n += 1
            if r.outcome == "resolved":
                ok += r.entity == g
                fm += r.entity != g
            amb += r.outcome == "ambiguous"
            cand += r.outcome == "candidates"
            unk += r.outcome == "unknown"
        res = ok + fm
        out.append(
            IdentityStats(
                world=world.name,
                policy=p.name,
                mentions=n,
                correct=ok,
                false_merges=fm,
                ambiguous=amb,
                candidates=cand,
                unknown=unk,
                false_merge_rate=Proportion.of(fm, res, CONFIDENCE),
                resolved_rate=Proportion.of(res, n, CONFIDENCE),
            )
        )
    return tuple(out)


def pool_extraction(stats: Sequence[ExtractionStats], world: str, policy: str) -> ExtractionStats:
    """Sum replicate statistics into one (counts add; proportions are recomputed)."""
    reasons: Counter[str] = Counter()
    for s in stats:
        reasons.update(dict(s.reasons))
    tot = {
        f: sum(getattr(s, f) for s in stats)
        for f in (
            "reports",
            "certain",
            "uncertain",
            "exact",
            "wrong_fact",
            "uncertain_as_fact",
            "missed",
            "unresolved_reports",
        )
    }
    v_ok = sum(s.claims_verified.numerator for s in stats)
    v_n = sum(s.claims_verified.denominator for s in stats)
    e, w = tot["exact"], tot["wrong_fact"]
    return ExtractionStats(
        world=world,
        policy=policy,
        **tot,
        reasons=tuple(sorted(reasons.items())),
        precision=Proportion.of(e, e + w, CONFIDENCE) if e + w else None,
        recall=Proportion.of(e, tot["certain"], CONFIDENCE) if tot["certain"] else None,
        conversion=(
            Proportion.of(tot["uncertain_as_fact"], tot["uncertain"], CONFIDENCE)
            if tot["uncertain"]
            else None
        ),
        claims_verified=Proportion.of(v_ok, v_n, CONFIDENCE),
    )


def pool_identity(stats: Sequence[IdentityStats], world: str, policy: str) -> IdentityStats:
    tot = {
        f: sum(getattr(s, f) for s in stats)
        for f in ("mentions", "correct", "false_merges", "ambiguous", "candidates", "unknown")
    }
    res = tot["correct"] + tot["false_merges"]
    return IdentityStats(
        world=world,
        policy=policy,
        **tot,
        false_merge_rate=Proportion.of(tot["false_merges"], res, CONFIDENCE),
        resolved_rate=Proportion.of(res, tot["mentions"], CONFIDENCE),
    )


__all__ = [
    "EXTRACTION_CASES",
    "IDENTITY_CASES",
    "REASONS",
    "ExtractionStats",
    "IdentityStats",
    "extraction_stats",
    "identity_stats",
    "merge_cases",
    "pool_extraction",
    "pool_identity",
    "run_extraction_cases",
    "run_identity_cases",
]
