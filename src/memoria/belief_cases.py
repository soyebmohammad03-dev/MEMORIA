"""Designed cases: the contradiction taxonomy and adversarial belief revision.

Small, hand-built evidence streams with known truth, run under every policy. They are
diagnostics, not samples: each isolates one way a belief system can be led astray.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import Field

from memoria.belief_lab import SELECTIVE, identity_map, with_signals
from memoria.beliefs import POLICIES, BeliefPolicy, EvidenceItem, evidence_from_statement
from memoria.comparison import normalize
from memoria.contradictions import (
    ClaimOntology,
    ContradictionType,
    classify,
    parse_assertion,
)
from memoria.core import EpistemicStatus, Record
from memoria.entities import CONSERVATIVE, resolve
from memoria.revision import RunMeta, covering, decide, run_ledger
from memoria.scenarios import day
from memoria.sources import SourceClass, SourceModel

DIGEST0 = "sha256:" + "0" * 64
ONTOLOGY = ClaimOntology(
    name="cases",
    version="1",
    hierarchy=(("berlin", "germany"), ("paris", "france")),
    categories=(("employer", ("acme", "globex", "initech")),),
    numeric_tolerance=0.02,
    concurrency_days=1.0,
    scopes=(("ana.home.summer", "ana.home"),),
)
SOURCES = SourceModel(
    name="cases",
    version="1",
    classes=(
        SourceClass(name="chat", alpha=7.5, beta=2.5),
        SourceClass(name="clinic", alpha=9.5, beta=0.5),
        SourceClass(name="email", alpha=8.5, beta=1.5),
        SourceClass(name="forum", alpha=4.0, beta=6.0),
    ),
    copies=tuple(sorted((f"bot:c{i}", "forum:o") for i in range(1, 6))),
)
_COLLISION = resolve({"ana": [DIGEST0], "anna": [DIGEST0]}, {}, CONSERVATIVE)


def _id(*parts: object) -> str:
    return "sha256:" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


# --- the contradiction taxonomy ------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    key: str
    raw: str
    day: float
    source: str
    verb: str = "set"


class TaxonomyCase(Record):
    name: str
    expected: ContradictionType | None  # None: compatible, no relation
    got: ContradictionType | None
    relation: str | None
    resolution: str | None
    genuine: bool | None
    ok: bool


def _case(name: str, a: Side, b: Side, expected: ContradictionType | None) -> TaxonomyCase:
    def parse(s: Side, n: str) -> object:
        when = day(s.day)
        return parse_assertion(_id(name, n), s.key, s.verb, s.raw, when, when, s.source)

    x, y = parse(a, "a"), parse(b, "b")
    r = classify(x, y, ONTOLOGY, SOURCES, _COLLISION)  # type: ignore[arg-type]
    got = r.type if r else None
    return TaxonomyCase(
        name=name,
        expected=expected,
        got=got,
        relation=r.relation if r else None,
        resolution=r.resolution if r else None,
        genuine=r.genuine if r else None,
        ok=got == expected,
    )


def taxonomy_cases() -> tuple[TaxonomyCase, ...]:
    s = Side
    table: list[tuple[str, Side, Side, ContradictionType | None]] = [
        (
            "direct value",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "Rome", 0, "chat:x2"),
            "direct_value",
        ),
        (
            "numeric",
            s("ana.dose", "10 mg", 0, "chat:x1"),
            s("ana.dose", "20 mg", 0.5, "chat:x2"),
            "numeric",
        ),
        (
            "categorical",
            s("ana.employer", "Acme", 0, "chat:x1"),
            s("ana.employer", "Globex", 0, "chat:x2"),
            "categorical",
        ),
        (
            "temporal (overlapping windows)",
            s("ana.home", "Paris (during=2026-01-01..2026-03-01)", 0, "chat:x1"),
            s("ana.home", "Rome (during=2026-02-01..2026-04-01)", 0, "chat:x2"),
            "temporal",
        ),
        (
            "negation",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "not Paris", 0.2, "chat:x2"),
            "negation",
        ),
        (
            "source disagreement",
            s("ana.home", "Paris", 0, "clinic:a1"),
            s("ana.home", "Rome", 0.1, "email:e1"),
            "source_disagreement",
        ),
        (
            "entity collision",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("anna.home", "Rome", 0.3, "chat:x2"),
            "entity_collision",
        ),
        (
            "scope",
            s("ana.home.summer", "Rome", 0, "chat:x1"),
            s("ana.home", "Paris", 0.2, "chat:x2"),
            "scope",
        ),
        (
            "partial",
            s("ana.home", "Paris; Berlin", 0, "chat:x1"),
            s("ana.home", "Paris; Rome", 0.4, "chat:x2"),
            "partial",
        ),
        (
            "supersession (change)",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "Berlin", 30, "chat:x2"),
            "supersession",
        ),
        (
            "supersession (correction)",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "Berlin", 5, "clinic:a1", "correct"),
            "supersession",
        ),
        (
            "granularity",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "France", 0.1, "chat:x2"),
            "granularity",
        ),
        (
            "missing qualifier",
            s("ana.dose", "10 mg (morning)", 0, "chat:x1"),
            s("ana.dose", "20 mg", 0.3, "chat:x2"),
            "missing_qualifier",
        ),
        (
            "compatible: same value",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.home", "paris", 0.1, "email:e1"),
            None,
        ),
        (
            "compatible: distinct qualifiers",
            s("ana.dose", "10 mg (morning)", 0, "chat:x1"),
            s("ana.dose", "20 mg (evening)", 0, "chat:x2"),
            None,
        ),
        (
            "compatible: not X with Y",
            s("ana.home", "not Paris", 0, "chat:x1"),
            s("ana.home", "Rome", 0.1, "chat:x2"),
            None,
        ),
        (
            "compatible: numbers within tolerance",
            s("ana.dose", "100 mg", 0, "chat:x1"),
            s("ana.dose", "101 mg", 0.1, "chat:x2"),
            None,
        ),
        (
            "compatible: other attribute",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("ana.employer", "Acme", 0.1, "chat:x2"),
            None,
        ),
        (
            "compatible: other entity, no candidate",
            s("ana.home", "Paris", 0, "chat:x1"),
            s("chen.home", "Rome", 0.1, "chat:x2"),
            None,
        ),
    ]
    return tuple(_case(*row) for row in table)


# --- adversarial cases ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    description: str
    key: str
    items: list[EvidenceItem]
    valid: float
    known: float
    truth: str
    sources: SourceModel = SOURCES
    withdrawals: list[tuple[datetime, str]] = field(default_factory=list)
    lineage_variants: bool = False  # run with and without lineage awareness
    identity: bool = False  # also run with identity-uncertainty damping


def _ev(
    case: str,
    n: int,
    key: str,
    value: str,
    occurred: float,
    recorded: float | None,
    source: str,
    verb: str = "set",
) -> EvidenceItem:
    rec = occurred if recorded is None else recorded
    content = f"{verb} {key} = {value}"
    item = evidence_from_statement(_id(case, n), content, day(occurred), day(rec), source)
    assert item is not None
    return item


def _derived(
    case: str,
    n: int,
    key: str,
    value: str,
    occurred: float,
    recorded: float,
    lineage: list[tuple[str, str]],
) -> EvidenceItem:
    return EvidenceItem(
        id=_id(case, "derived", n),
        key=key,
        verb="set",
        value=" ".join(normalize(value)),
        raw=value,
        occurred_at=day(occurred),
        recorded_at=day(recorded),
        source=f"derived:m{n}",
        status=EpistemicStatus.DERIVED,
        multiplicity=len(lineage),
        lineage=tuple(sorted(lineage)),
    )


def adversarial_cases() -> list[Case]:
    k = "ana.home"
    out: list[Case] = []

    def add(
        name: str,
        description: str,
        items: list[EvidenceItem],
        valid: float,
        known: float,
        truth: str,
        **kw: object,
    ) -> None:
        out.append(Case(name, description, k, items, valid, known, truth.lower(), **kw))  # type: ignore[arg-type]

    n = "weak-flood"
    items = [_ev(n, 0, k, "Paris", 1, None, "clinic:a1")]
    items += [_ev(n, i, k, "Rome", 1 + 0.1 * i, None, f"forum:f{i}") for i in range(1, 7)]
    add(n, "six independent weak sources against one strong source", items, 20, 30, "Paris")

    n = "copied-evidence"
    items = [_ev(n, 0, k, "Rome", 1, None, "forum:o")]
    items += [_ev(n, i, k, "Rome", 1 + 0.01 * i, None, f"bot:c{i}") for i in range(1, 6)]
    items.append(_ev(n, 9, k, "Paris", 1.3, None, "email:e1"))
    add(
        n,
        "one wrong report copied five times against one independent report",
        items,
        20,
        30,
        "Paris",
    )

    n = "recent-contradiction"
    items = [
        _ev(n, 1, k, "Paris", 0, None, "clinic:a1"),
        _ev(n, 2, k, "Paris", 0.3, None, "email:e1"),
        _ev(n, 3, k, "Paris", 0.5, None, "email:e2"),
        _ev(n, 4, k, "Rome", 40, None, "forum:f1"),
    ]
    add(n, "a single recent weak report against three established ones", items, 45, 50, "Paris")

    n = "stale-confidence"
    items = [
        _ev(n, i, k, "Paris", 0.5 * i, None, s)
        for i, s in enumerate(["clinic:a1", "email:e1", "email:e2", "chat:c1"])
    ]
    items.append(_ev(n, 5, k, "Berlin", 100, None, "clinic:a1"))
    add(
        n,
        "an established value, then a true change reported by one strong source",
        items,
        110,
        112,
        "Berlin",
    )

    n = "entity-collision"
    items = [
        _ev(n, 1, k, "Paris", 0, None, "clinic:a1"),
        _ev(n, 2, k, "Paris", 0.4, None, "email:e1"),
        _ev(n, 3, k, "Paris", 0.6, None, "chat:c1"),
        _ev(n, 4, "anna.home", "Rome", 0.2, None, "email:e2"),
        _ev(n, 5, k, "Rome", 0.3, None, "chat:c2"),
    ]
    add(
        n,
        "a report about a near-collision entity filed under this one",
        items,
        20,
        30,
        "Paris",
        identity=True,
    )

    n = "temporal-spoofing"
    items = [
        _ev(n, 1, k, "Paris", 0, None, "clinic:a1"),
        _ev(n, 2, k, "Rome", 3, 200, "forum:f1"),
        _ev(n, 3, k, "Paris", 150, None, "clinic:a1"),
    ]
    add(n, "a backdated report that rewrites an interval already covered", items, 50, 210, "Paris")

    n = "reputation-poisoning"
    items = [_ev(n, i, k, "Paris", 20 * i, None, "email:trusted") for i in range(6)]
    items += [
        _ev(n, 20, k, "Rome", 130, None, "email:trusted"),
        _ev(n, 21, k, "Rome", 130.2, None, "forum:f1"),
        _ev(n, 22, k, "Paris", 130.4, None, "clinic:a1"),
    ]
    add(n, "a source that earned trust with true reports turns and lies", items, 135, 140, "Paris")

    n = "alternating"
    items = [_ev(n, 0, k, "Paris", 0, None, "clinic:a1")]
    items += [_ev(n, 2 * i + 1, k, "Paris", 3 + 4 * i, None, "chat:c1") for i in range(8)]
    items += [_ev(n, 2 * i + 2, k, "Rome", 5 + 4 * i, None, "chat:c2") for i in range(8)]
    add(n, "two equal sources alternate between two values every two days", items, 30, 40, "Paris")

    n = "misleading-summary"
    copies = [(_id(n, "x", i), "forum:o" if i == 0 else f"bot:c{i}") for i in range(5)]
    items = [_derived(n, 1, k, "Rome", 1, 50, copies), _ev(n, 9, k, "Paris", 1.3, None, "email:e1")]
    add(
        n,
        "a summary of five copies of one report counted as five",
        items,
        60,
        70,
        "Paris",
        lineage_variants=True,
    )

    n = "weakened-derived"
    raw = [
        _ev(n, i, k, "Paris", 1 + 0.1 * i, None, s)
        for i, s in enumerate(["clinic:a1", "email:e1", "chat:c1"])
    ]
    lineage = [(i.id, i.source) for i in raw]
    add(
        n,
        "a summary outlives all the evidence it summarised",
        [*raw, _derived(n, 1, k, "Paris", 1, 30, lineage)],
        60,
        70,
        "Paris",
        withdrawals=[(day(40), i.id) for i in raw],
        lineage_variants=True,
    )
    return out


class CaseResult(Record):
    case: str
    policy: str
    lineage: bool
    identity: bool = False
    mode: str
    value: str | None
    score: float | None
    correct: bool | None
    overconfident: bool  # answered wrong with score >= 0.6, or confident with nothing left
    state: str | None
    independent: int | None
    naive: int | None
    events: int
    flips: int = Field(default=0)


def run_case(
    c: Case, policy: BeliefPolicy, *, lineage: bool = False, identity: bool = False
) -> CaseResult:
    pol = with_signals(policy, "lineage") if lineage else policy
    if identity:
        pol = with_signals(pol, "identity")
    ident = identity_map([c.key, "anna.home"]) if identity else {}
    # Withdrawn lineage experiences are not evidence items themselves; give the ledger
    # the raw items and the derived item together.
    run = run_ledger(
        c.items,
        pol,
        c.sources,
        identity=ident,
        withdrawals=c.withdrawals,
        meta=RunMeta(world=c.name),
    )
    valid, known = day(c.valid), day(c.known)
    kb = run.at(c.key, known)
    d = decide(c.key, kb, valid, known, SELECTIVE)
    cover = covering(kb, valid)
    chosen = next((b for b in cover if b.id in d.beliefs), None)
    answered = d.mode in ("answer", "answer_with_uncertainty")
    correct = (d.values[0] == c.truth) if answered and d.values else None
    score = d.score if chosen is not None else None
    confident = answered and score is not None and score >= 0.6
    gone = {e for t, e in c.withdrawals if t <= known}
    by_id = {i.id: i for i in c.items}
    nothing_left = bool(
        confident
        and chosen is not None
        and chosen.supporting
        and all(
            by_id[e].lineage and {x for x, _ in by_id[e].lineage} <= gone for e in chosen.supporting
        )
    )
    values = [
        s.beliefs and next((b.value for b in s.beliefs if b.state.value == "supported"), None)
        for s in run.history.get(c.key, [])
    ]
    seq = [v for v in values if v]
    flips = sum(1 for a, b in itertools.pairwise(seq) if a != b)
    return CaseResult(
        case=c.name,
        policy=policy.name,
        lineage=lineage,
        identity=identity,
        mode=d.mode,
        value=d.values[0] if answered and d.values else None,
        score=score,
        correct=correct,
        overconfident=bool(confident and (correct is False or nothing_left)),
        state=chosen.state.value if chosen else None,
        independent=chosen.independent_count if chosen else None,
        naive=chosen.naive_count if chosen else None,
        events=len(run.ledger.events),
        flips=flips,
    )


def run_cases() -> tuple[CaseResult, ...]:
    results = []
    for c in adversarial_cases():
        for name in sorted(POLICIES):
            for lin in (False, True) if c.lineage_variants else (False,):
                for ident in (False, True) if c.identity else (False,):
                    results.append(run_case(c, POLICIES[name], lineage=lin, identity=ident))
    return tuple(results)
