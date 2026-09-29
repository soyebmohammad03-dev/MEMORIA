"""Memory autopsy, temporal replay and counterfactual replay (Super-Phase 6).

**Replay.** A finished run stores everything a decision used: the recorded items and the event
ledger. ``replay_memory`` rebuilds the memory *as of a cutoff* into fresh objects (items
recorded at or before it, events recorded strictly before it) and applies the same decision
code the simulator used (:class:`memoria.retention.Memory`). It never touches the live run
(the ledger is copied, so even the freeze on decisions does not move), and
:func:`historical_mismatches` checks that replaying at every recorded decision time reproduces
the answer the live run gave.

**Autopsy.** :func:`memory_autopsy` explains one answer at one time as an ordered chain of
:class:`Link`s (answer, decision, ranked candidates, signals, policy, selected version,
retention and importance state, entity and claim resolution, originating experience, source,
temporal validity, corrections, contradictions), each ``found``, ``missing`` (with what is
missing) or ``not_applicable`` (with why), plus what happened *after* the cutoff (later
evidence, events and answers), which the decision never saw. Every reference resolves in the
stored run or is reported missing; :func:`verify_autopsy` rebuilds the autopsy and checks the
digest and the references. Hidden truth appears only in ``analysis``. :func:`belief_autopsy`
maps a Phase 11-12 belief trace onto the same chain.

**Counterfactual.** :func:`counterfactual` recomputes every recorded decision with exactly one
of the schedule, rank weight or importance model changed while the evidence (items and event
ledger) stays byte-identical, and reports which decisions changed and the recorded reason. It
is a decision-level intervention on identical inputs, *not* a causal estimate of what the
system would have done under the alternative: its own feedback events would have differed. A
loop-inclusive re-simulation is run beside it to show how much they diverge.

Replay of the Phase 1-7 substrate (:func:`substrate_state`, :func:`substrate_hierarchy`) is a
thin wrapper over the existing pure functions, with a digest of the log to show it is unchanged.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field

from memoria.adaptive import compute, score
from memoria.adaptive_worlds import Report, World, truth_at, truth_index
from memoria.claims import EXTRACTOR, Claim, verify_claim
from memoria.consolidation import ConsolidationPolicy, Hierarchy, consolidate
from memoria.core import Digest, MemoryState, Record, content_hash, quantize
from memoria.embeddings import Embedder
from memoria.hybrid import Corpus
from memoria.retention import (
    CONCURRENCY_DAYS,
    Item,
    Memory,
    SimResult,
    System,
    simulate,
)
from memoria.statistics import Proportion
from memoria.store import MemoryLog

Status = Literal["found", "missing", "not_applicable"]
MEMORY_LINKS = (
    "answer",
    "decision",
    "candidates",
    "policy",
    "signals",
    "selected",
    "state",
    "resolution",
    "experience",
    "source",
    "validity",
    "corrections",
    "contradictions",
)
INTERPRETATION = (
    "decision-level recomputation with the recorded evidence held fixed; not a causal estimate "
    "of system behaviour under the alternative, whose own feedback events would differ"
)


# --- replay -------------------------------------------------------------------------------


def replay_memory(
    result: SimResult, system: System, cutoff: float, *, truncate: bool = True
) -> Memory:
    """The memory of ``system`` as of ``cutoff``, rebuilt from the stored run.

    With ``truncate`` only items recorded at or before the cutoff and events recorded strictly
    before it are copied in, so nothing later can reach a decision; without it everything is
    copied and the decision functions apply the same cutoff themselves (used for bulk replay).
    """
    assert result.ledger is not None
    mem = Memory(system, result.ledger.copy(cutoff if truncate else None))
    for it in sorted(result.catalog.values(), key=lambda i: (i.recorded, i.id)):
        if not truncate or it.recorded <= cutoff:
            mem.by_key[it.key].append(it)
    return mem


def ordered(mem: Memory, cands: Sequence[Item], t: float) -> list[Item]:
    """Candidates best first, by the exact ordering ``Memory.choose`` maximises."""
    if mem.system.rank_weight == 0:
        return sorted(cands, key=lambda i: (i.occurred, i.recorded, i.id), reverse=True)
    return sorted(cands, key=lambda i: mem.rank_key(i, t), reverse=True)


class StateSnapshot(Record):
    """What a system holds and answers at one cutoff. Its digest identifies the state."""

    cutoff: float
    system: Digest
    items_known: int
    events_visible: int
    answers: tuple[tuple[str, str | None], ...]  # key -> selected item
    states: tuple[tuple[str, str, str], ...]  # item, active|archived, reason
    stale_exposed: tuple[
        str, ...
    ]  # keys answered by an item a known, later, different one supersedes
    contested: tuple[str, ...]  # keys with concurrent disagreement among known items


def superseded_by(mem: Memory, it: Item, t: float) -> list[str]:
    """Known items of the same key with another value that occurred clearly later. Evidence
    only: "later" is not "truer", and hidden truth is not consulted."""
    return sorted(
        j.id
        for j in mem.known(it.key, t)
        if j.value != it.value and j.occurred > it.occurred + CONCURRENCY_DAYS
    )


def contested(mem: Memory, key: str, t: float) -> bool:
    known = mem.known(key, t)
    if not known:
        return False
    top = max(i.occurred for i in known)
    return len({i.value for i in known if i.occurred >= top - CONCURRENCY_DAYS}) > 1


def snapshot(
    result: SimResult, system: System, cutoff: float, keys: Sequence[str] | None = None
) -> StateSnapshot:
    mem = replay_memory(result, system, cutoff)
    ks = sorted(keys if keys is not None else mem.by_key)
    answers, states, stale, cont = [], [], [], []
    for k in ks:
        dec = mem.decisions(k, cutoff)
        states += [(it.id, "active" if ok else "archived", why) for it, ok, why, _ in dec]
        chosen = mem.choose([it for it, ok, _, _ in dec if ok], cutoff)
        answers.append((k, chosen.id if chosen else None))
        if chosen is not None and superseded_by(mem, chosen, cutoff):
            stale.append(k)
        if contested(mem, k, cutoff):
            cont.append(k)
    return StateSnapshot(
        cutoff=cutoff,
        system=system.digest,
        items_known=sum(len(v) for v in mem.by_key.values()),
        events_visible=mem.ledger.count,
        answers=tuple(answers),
        states=tuple(sorted(states)),
        stale_exposed=tuple(stale),
        contested=tuple(cont),
    )


def historical_mismatches(
    result: SimResult, system: System, every: int = 1
) -> tuple[int, list[str]]:
    """(decisions checked, mismatches): replaying the stored run at each recorded decision time
    (every ``every``-th decision) must give the answer the live run gave; a mismatch means a
    decision depended on something the stored records do not hold."""
    bad: list[str] = []
    checked = 0
    mems: dict[float, Memory] = {}
    points = [("audit", a.t, a.key, a.item) for a in result.audits] + [
        ("query", t, key, item) for _, t, key, item in result.query_log
    ]
    for n, (kind, t, key, item) in enumerate(sorted(points, key=lambda p: (p[1], p[2], p[0]))):
        if n % every:
            continue
        mem = mems.get(t) or replay_memory(result, system, t)
        mems[t] = mem
        again = mem.choose(mem.available(key, t), t)
        checked += 1
        if (again.id if again else None) != item:
            bad.append(f"{kind} {key}@{t:g}")
    return checked, bad


def ledger_digest(result: SimResult) -> str:
    """Identity of the stored evidence: item ids and values and every event id."""
    assert result.ledger is not None
    return content_hash(
        [
            [
                (i.id, i.key, i.value, i.occurred, i.recorded)
                for _, i in sorted(result.catalog.items())
            ],
            [e.id for e in result.ledger.all()],
        ]
    )


# --- autopsy ------------------------------------------------------------------------------


class Link(Record):
    step: str
    status: Status
    detail: tuple[tuple[str, str], ...] = ()
    refs: tuple[str, ...] = ()  # ids of records that justify this link


class Ranked(Record):
    item: str
    value: str
    state: Literal["active", "archived"]
    reason: str
    basis: tuple[str, ...]
    freshness: float | None
    importance: float | None
    rank_score: float | None
    rank: int | None  # 1 = selected; None: not retrievable


class Autopsy(Record):
    """The evidence chain behind one answer or belief at one cutoff."""

    subject: Literal["memory", "belief"]
    world: Digest
    system: str  # digest of the memory system, or the belief policy name
    key: str
    at: float  # the cutoff (days)
    kind: Literal["audit", "query", "adhoc"]
    answer: str | None
    candidates: tuple[Ranked, ...]
    links: tuple[Link, ...]
    after_cutoff: tuple[Link, ...]  # what later happened; never seen by the decision
    analysis: tuple[tuple[str, str], ...]  # hidden truth: for analysis, not part of the chain
    missing: tuple[str, ...]
    complete: bool = Field(default=False)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.6f}"


def memory_autopsy(
    world: World,
    system: System,
    result: SimResult,
    claims: Mapping[str, tuple[Claim, ...]] | None,
    key: str,
    at: float,
    kind: Literal["audit", "query", "adhoc"] = "adhoc",
    reports: Mapping[str, Report] | None = None,
) -> Autopsy:
    """Explain what ``system`` answers for ``key`` at ``at``, from the stored run only.

    ``reports`` defaults to the world's reports; passing fewer shows how a missing originating
    experience is reported rather than guessed."""
    assert result.ledger is not None
    rep = {r.id: r for r in world.reports} if reports is None else dict(reports)
    mem = replay_memory(result, system, at)
    spec = system.importance
    dec = mem.decisions(key, at)
    cands_active = [it for it, ok, _, _ in dec if ok]
    order = ordered(mem, cands_active, at)
    chosen = mem.choose(cands_active, at)
    lam = system.rank_weight
    rows: list[Ranked] = []
    rank_of = {it.id: n + 1 for n, it in enumerate(order)}
    for it, ok, why, basis in sorted(dec, key=lambda d: (rank_of.get(d[0].id, 10**9), d[0].id)):
        fresh = 0.5 ** (max(0.0, at - it.occurred) / spec.half_life_days)
        imp = score(mem.ledger, it.id, it.recorded, it.prior, at, spec)
        rows.append(
            Ranked(
                item=it.id,
                value=it.value,
                state="active" if ok else "archived",
                reason=why,
                basis=basis,
                freshness=quantize(fresh),
                importance=imp,
                rank_score=quantize((1 - lam) * fresh + lam * imp) if ok else None,
                rank=rank_of.get(it.id),
            )
        )
    missing: list[str] = []
    links: list[Link] = []

    def add(
        step: str,
        status: Status,
        detail: Sequence[tuple[str, str]] = (),
        refs: Sequence[str] = (),
        why: str = "",
    ) -> None:
        links.append(Link(step=step, status=status, detail=tuple(detail), refs=tuple(refs)))
        if status == "missing":
            missing.append(f"{step}: {why}")

    add(
        "answer",
        "found",
        [
            ("value", chosen.value if chosen else "-"),
            ("item", chosen.id if chosen else "-"),
            ("kind", kind),
        ],
        [chosen.id] if chosen else [],
    )
    add(
        "decision",
        "found",
        [
            ("rule", "argmax (1 - lambda) * freshness + lambda * importance"),
            ("lambda", f"{lam:g}"),
            ("known", str(len(dec))),
            ("available", str(len(cands_active))),
        ],
        [],
    )
    add(
        "candidates",
        "found",
        [(f"#{r.rank}" if r.rank else "archived", f"{r.item} {r.value}") for r in rows],
        [r.item for r in rows],
    )
    add(
        "policy",
        "found",
        [
            ("system", system.digest),
            ("schedule", system.schedule.digest),
            ("importance", spec.digest),
            ("extractor", EXTRACTOR.digest),
            ("identity", system.identity),
            ("ingest", system.ingest),
        ],
        [],
    )
    if chosen is None:
        for step in ("signals", *MEMORY_LINKS[5:]):
            add(step, "missing", why="no retrievable item answers this key at the cutoff")
        return _finish(world, system, key, at, kind, None, rows, links, [], result, mem, missing)
    rec = compute(mem.ledger, chosen.id, chosen.recorded, chosen.prior, at, spec)
    add(
        "signals",
        "found",
        [(c.name, f"{c.value:.6f} x {c.weight:g} raw={dict(c.raw)}") for c in rec.components]
        + [("importance", _fmt(rec.score)), ("importance_digest", rec.digest)],
        list(rec.events),
    )
    add(
        "selected",
        "found",
        [
            ("item", chosen.id),
            ("key", chosen.key),
            ("value", chosen.value),
            ("kind", chosen.kind),
            ("recorded", _fmt(chosen.recorded)),
        ],
        [chosen.id],
    )
    row = next(r for r in rows if r.item == chosen.id)
    add(
        "state",
        "found",
        [
            ("retention", row.reason),
            ("schedule", system.schedule.name),
            ("importance", _fmt(rec.score)),
            ("basis", ",".join(row.basis) or "-"),
        ],
        list(row.basis) + list(rec.events),
    )
    # resolution
    if chosen.claim is None:
        add("resolution", "not_applicable", [("why", "gold ingestion has no extraction step")])
    else:
        c = next((c for cs in (claims or {}).values() for c in cs if c.id == chosen.claim), None)
        r0 = rep.get(chosen.report)
        if c is None:
            add("resolution", "missing", why=f"claim {chosen.claim} is not in the extraction")
        else:
            problems = verify_claim(c, r0.text) if r0 else ["text unavailable"]
            add(
                "resolution",
                "found" if not problems else "missing",
                [
                    ("claim", c.id),
                    ("rule", c.rule or "-"),
                    ("entity", c.entity or "-"),
                    ("attribute", c.attribute or "-"),
                    ("value", c.value or "-"),
                    ("span_verified", str(not problems)),
                ],
                [c.id],
                why="; ".join(problems),
            )
    r0 = rep.get(chosen.report)
    if r0 is None:
        add("experience", "missing", why=f"originating report {chosen.report} is not available")
    else:
        add(
            "experience",
            "found",
            [
                ("report", r0.id),
                ("text", r0.text),
                ("occurred", _fmt(r0.occurred)),
                ("recorded", _fmt(r0.recorded)),
            ],
            [r0.id],
        )
    cls = chosen.source.partition(":")[0]
    prior = dict(world.declared).get(cls)
    if prior is None:
        add("source", "missing", why=f"no declared prior for source class {cls!r}")
    else:
        add(
            "source",
            "found",
            [("source", chosen.source), ("class", cls), ("declared_prior", f"{prior:g}")],
            [chosen.source],
        )
    sup = superseded_by(mem, chosen, at)
    add(
        "validity",
        "found",
        [
            ("occurred", _fmt(chosen.occurred)),
            ("recorded", _fmt(chosen.recorded)),
            ("superseded_by_known", ",".join(sup) or "none"),
        ],
        sup,
    )
    seen = mem.ledger.visible(chosen.id, at)
    cor = [e for e in seen if e.kind == "correction"]
    con = [e for e in seen if e.kind == "contradiction_discovered"]
    add(
        "corrections",
        "found",
        [(e.id, f"{e.origin} at {e.recorded_at:g} cause {e.cause}") for e in cor]
        or [("count", "0")],
        [e.id for e in cor],
    )
    add(
        "contradictions",
        "found",
        [(e.id, f"{e.origin} at {e.recorded_at:g} cause {e.cause}") for e in con]
        or [("count", "0")],
        [e.id for e in con],
    )
    # after the cutoff: recorded in the stored run, never visible to this decision
    later_items = sorted(
        (i for i in result.catalog.values() if i.key == key and i.recorded > at),
        key=lambda i: (i.recorded, i.id),
    )
    later_events = [e for e in result.ledger.all() if e.item == chosen.id and e.recorded_at >= at]
    later_answers = [a for a in result.audits if a.key == key and a.t > at]
    after = [
        Link(
            step="later_evidence",
            status="found",
            detail=tuple((i.id, f"{i.value} recorded {i.recorded:g}") for i in later_items[:20])
            or (("count", "0"),),
            refs=tuple(i.id for i in later_items[:20]),
        ),
        Link(
            step="later_events",
            status="found",
            detail=tuple(
                (e.id, f"{e.kind} ({e.origin}) recorded {e.recorded_at:g}")
                for e in later_events[:20]
            )
            or (("count", "0"),),
            refs=tuple(e.id for e in later_events[:20]),
        ),
        Link(
            step="later_answers",
            status="found",
            detail=tuple((f"{a.t:g}", a.item or "-") for a in later_answers[:20])
            or (("count", "0"),),
            refs=tuple(a.item for a in later_answers[:20] if a.item),
        ),
    ]
    return _finish(world, system, key, at, kind, chosen, rows, links, after, result, mem, missing)


def _finish(
    world: World,
    system: System,
    key: str,
    at: float,
    kind: Literal["audit", "query", "adhoc"],
    chosen: Item | None,
    rows: list[Ranked],
    links: list[Link],
    after: list[Link],
    result: SimResult,
    mem: Memory,
    missing: list[str],
) -> Autopsy:
    truth, _ = truth_at(truth_index(world), key, at)
    status = (
        "none"
        if chosen is None
        else "correct"
        if chosen.value == truth
        else "not the current truth"
    )
    return Autopsy(
        subject="memory",
        world=world.digest,
        system=system.digest,
        key=key,
        at=at,
        kind=kind,
        answer=chosen.value if chosen else None,
        candidates=tuple(rows),
        links=tuple(links),
        after_cutoff=tuple(after),
        analysis=(("truth", truth), ("status", status)),
        missing=tuple(missing),
        complete=not missing,
    )


def verify_autopsy(
    a: Autopsy,
    world: World,
    system: System,
    result: SimResult,
    claims: Mapping[str, tuple[Claim, ...]] | None,
) -> list[str]:
    """Rebuild the autopsy from the stored run and check it: same digest, and every reference
    resolves to an item, event, report, claim or source of the run (empty: no guesses)."""
    assert result.ledger is not None
    bad: list[str] = []
    again = memory_autopsy(world, system, result, claims, a.key, a.at, a.kind)
    if again.digest != a.digest:
        bad.append("rebuilding the autopsy gives a different record")
    items = set(result.catalog)
    events = {e.id for e in result.ledger.all()}
    reports = {r.id for r in world.reports}
    claim_ids = {c.id for cs in (claims or {}).values() for c in cs}
    sources = {i.source for i in result.catalog.values()}
    for link in (*a.links, *a.after_cutoff):
        for ref in link.refs:
            ok = (
                ref in items
                or ref in events
                or ref in reports
                or ref in claim_ids
                or ref in sources
            )
            if not ok:
                bad.append(f"{link.step}: reference {ref} does not resolve")
    return bad


# --- belief autopsy -----------------------------------------------------------------------


def belief_autopsy(
    world_digest: str,
    policy_name: str,
    trace: object,
    later: Sequence[str],
    key: str,
    known_day: float,
    answer: str | None,
    candidates: Sequence[tuple[str, str, str, float]],
) -> Autopsy:
    """Map a :class:`memoria.belief_demo.BeliefTrace` onto the autopsy chain.

    ``candidates``: (belief id, value, state, score) of every belief of the key at the cutoff;
    ``later``: descriptions of revision events after the cutoff. The trace's own ``missing``
    list becomes the chain's missing steps; nothing is inferred beyond it."""
    from memoria.belief_demo import BeliefTrace

    assert isinstance(trace, BeliefTrace)
    missing = list(trace.missing)
    rows = tuple(
        Ranked(
            item=b,
            value=v,
            state="active",
            reason=st,
            basis=(),
            freshness=None,
            importance=None,
            rank_score=sc,
            rank=n + 1,
        )
        for n, (b, v, st, sc) in enumerate(sorted(candidates, key=lambda c: (-c[3], c[0])))
    )

    def has(sub: str) -> bool:
        return any(sub in m for m in missing)

    links = [
        Link(
            step="answer",
            status="found",
            detail=(("mode", trace.decision.mode), ("values", ",".join(trace.decision.values))),
        ),
        Link(
            step="decision",
            status="found",
            detail=(
                ("policy", policy_name),
                ("score", _fmt(trace.score)),
                ("arithmetic", trace.arithmetic),
            ),
        ),
        Link(step="policy", status="found", detail=(("belief_policy", policy_name),)),
        Link(
            step="candidates",
            status="found",
            detail=tuple((r.item, f"{r.value} {r.reason} {r.rank_score}") for r in rows),
            refs=tuple(r.item for r in rows),
        ),
    ]
    if trace.belief is None:
        links += [
            Link(step=s, status="missing", detail=(("why", "no belief answers this query"),))
            for s in (
                "signals",
                "selected",
                "state",
                "resolution",
                "experience",
                "source",
                "validity",
                "corrections",
                "contradictions",
            )
        ]
        return Autopsy(
            subject="belief",
            world=world_digest,
            system=policy_name,
            key=key,
            at=known_day,
            kind="adhoc",
            answer=answer,
            candidates=rows,
            links=tuple(links),
            after_cutoff=(),
            analysis=(),
            missing=tuple(missing),
            complete=False,
        )
    ev = trace.evidence
    links += [
        Link(
            step="signals",
            status="found" if trace.arithmetic == "ok" else "missing",
            detail=(
                ("weight", _fmt(trace.weight)),
                ("mass", _fmt(trace.mass)),
                ("prior_mass", _fmt(trace.prior_mass)),
                *((e.id, ",".join(f"{k}={v:.3f}" for k, v in e.factors)) for e in ev if e.factors),
            ),
            refs=tuple(e.id for e in ev),
        ),
        Link(
            step="selected",
            status="found",
            detail=(("belief", trace.belief), ("state", trace.state.value if trace.state else "-")),
            refs=(trace.belief,),
        ),
        Link(
            step="state",
            status="found",
            detail=tuple(
                (f"event {e.seq}", f"{e.reason}: {'; '.join(e.changes)}") for e in trace.events
            )
            + tuple((e.id, ",".join(e.derived_by) or "not consolidated") for e in ev),
            refs=(),
        ),
        Link(
            step="resolution",
            status="missing" if has("no graph claim") else "found",
            detail=tuple((e.id, e.graph_claim or "-") for e in ev),
            refs=tuple(e.graph_claim for e in ev if e.graph_claim),
        ),
        Link(
            step="experience",
            status="missing"
            if has("experience not in the corpus") or has("no memory version")
            else "found",
            detail=tuple((e.id, e.content or "-") for e in ev),
            refs=tuple(e.id for e in ev),
        ),
        Link(
            step="source",
            status="found",
            detail=tuple((e.id, f"{e.source} root {e.root}") for e in ev),
            refs=tuple(sorted({e.source for e in ev})),
        ),
        Link(
            step="validity",
            status="found",
            detail=(
                ("valid_at", str(trace.valid_at)),
                ("known_at", str(trace.known_at)),
                *((e.id, f"occurred {e.occurred_at} recorded {e.recorded_at}") for e in ev),
            ),
        ),
        Link(
            step="corrections",
            status="found",
            detail=tuple(
                (f"event {e.seq}", e.reason) for e in trace.events if "correct" in e.reason
            )
            or (("count", "0"),),
        ),
        Link(
            step="contradictions",
            status="found",
            detail=tuple((e.id, e.role) for e in ev if e.role == "contradicting")
            or (("count", "0"),),
            refs=tuple(e.id for e in ev if e.role == "contradicting"),
        ),
    ]
    after = (
        Link(
            step="later_events",
            status="found",
            detail=tuple((str(n), d) for n, d in enumerate(later)) or (("count", "0"),),
        ),
    )
    return Autopsy(
        subject="belief",
        world=world_digest,
        system=policy_name,
        key=key,
        at=known_day,
        kind="adhoc",
        answer=answer,
        candidates=rows,
        links=tuple(links),
        after_cutoff=after,
        analysis=(),
        missing=tuple(missing),
        complete=not missing,
    )


def belief_context(
    run: object, key: str, known_at: datetime
) -> tuple[list[tuple[str, str, str, float]], list[str]]:
    """The candidates (belief id, value, state, score) of ``key`` at ``known_at`` and the
    descriptions of revision events for the key recorded *after* it, from a belief ledger run."""
    from memoria.revision import LedgerRun

    assert isinstance(run, LedgerRun)
    kb = run.at(key, known_at)
    cands = [(b.id, b.value, b.state.value, b.score) for b in (kb.beliefs if kb else ())]
    last = kb.seq if kb else -1
    later = [f"seq {e.seq} {e.reason}" for e in run.ledger.events if e.key == key and e.seq > last]
    return cands, later


# --- counterfactual -----------------------------------------------------------------------


class Why(Record):
    kind: Literal["retention", "ranking", "importance"]
    detail: tuple[tuple[str, str], ...]


class ChangedDecision(Record):
    kind: Literal["audit", "query"]
    at: float
    key: str
    base_item: str | None
    alt_item: str | None
    base_value: str | None
    alt_value: str | None
    why: Why
    base_status: str  # analysis only: against hidden truth
    alt_status: str


class Counterfactual(Record):
    """One system against one alternative on the same, unchanged evidence."""

    world: Digest
    base: Digest
    alt: Digest
    changed_field: Literal["schedule", "rank_weight", "importance"]
    evidence: Digest  # items and events: identical before and after (checked)
    decisions: int
    changed: int
    changed_share: Proportion
    by_why: tuple[tuple[str, int], ...]
    changes: tuple[ChangedDecision, ...]
    gained: int  # audit decisions the alternative gets right that the base got wrong (analysis)
    lost: int
    resimulated_changed: int  # audit answers that differ when the alternative is re-simulated
    fixed_vs_resimulated: int  # audit answers where the fixed-ledger and re-simulated alt differ
    interpretation: str


def _changed_field(base: System, alt: System) -> Literal["schedule", "rank_weight", "importance"]:
    diff = [
        f for f in ("schedule", "rank_weight", "importance") if getattr(base, f) != getattr(alt, f)
    ]
    others = [
        f
        for f in ("ingest", "identity", "feedback_lag_days")
        if getattr(base, f) != getattr(alt, f)
    ]
    if len(diff) != 1 or others:
        raise ValueError(
            "a counterfactual changes exactly one of schedule, rank_weight, importance; "
            f"got {[*diff, *others]}"
        )
    return diff[0]  # type: ignore[return-value]


def _status(
    world_index: dict[str, tuple[list[float], list[str]]], it: Item | None, key: str, t: float
) -> str:
    truth, period = truth_at(world_index, key, t)
    if it is None:
        return "none"
    if it.value == truth:
        return "correct"
    return "stale" if it.value in set(world_index[key][1][:period]) else "wrong"


def _explain(
    base_mem: Memory,
    alt_mem: Memory,
    b: Item | None,
    a: Item | None,
    key: str,
    t: float,
    field: str,
) -> Why:
    """Why the answer changed: the recorded state of the two winners under both systems."""
    detail: list[tuple[str, str]] = [
        ("base_winner", b.id if b else "-"),
        ("alt_winner", a.id if a else "-"),
    ]
    if b is not None:
        d = {it.id: (ok, why, basis) for it, ok, why, basis in alt_mem.decisions(key, t)}
        ok, why, basis = d[b.id]
        if not ok:
            detail += [
                ("base_winner_under_alt", f"archived: {why}"),
                ("basis", ",".join(basis) or "-"),
            ]
            return Why(kind="retention", detail=tuple(detail))
    for label, mem in (("base", base_mem), ("alt", alt_mem)):
        lam = mem.system.rank_weight
        for name, it in (("base_winner", b), ("alt_winner", a)):
            if it is None:
                continue
            fresh = 0.5 ** (max(0.0, t - it.occurred) / mem.system.importance.half_life_days)
            imp = score(mem.ledger, it.id, it.recorded, it.prior, t, mem.system.importance)
            detail.append(
                (
                    f"{label}:{name}",
                    f"freshness {fresh:.6f} importance {imp:.6f} lambda {lam:g} -> "
                    f"{quantize((1 - lam) * fresh + lam * imp):.6f}",
                )
            )
    return Why(kind="importance" if field == "importance" else "ranking", detail=tuple(detail))


def counterfactual(
    world: World,
    base: System,
    alt: System,
    base_result: SimResult,
    claims: dict[str, tuple[Claim, ...]] | None = None,
) -> Counterfactual:
    """Recompute every decision of the run with exactly one policy component changed and the
    evidence fixed; see the module docstring for what this does and does not show."""
    assert base_result.ledger is not None
    field = _changed_field(base, alt)
    before = ledger_digest(base_result)
    live_frozen = base_result.ledger.frozen
    index = truth_index(world)
    alt_mem = replay_memory(base_result, alt, 0.0, truncate=False)
    base_mem = replay_memory(base_result, base, 0.0, truncate=False)
    points: list[tuple[Literal["audit", "query"], float, str, str | None]] = [
        ("audit", a.t, a.key, a.item) for a in base_result.audits
    ] + [("query", t, k, item) for _, t, k, item in base_result.query_log]
    points.sort(key=lambda p: (p[1], p[2], p[0]))
    changes: list[ChangedDecision] = []
    gained = lost = 0
    alt_audit: dict[tuple[float, str], str | None] = {}
    for kind, t, key, base_item in points:
        alt_pick = alt_mem.choose(alt_mem.available(key, t), t)
        alt_id = alt_pick.id if alt_pick else None
        if kind == "audit":
            alt_audit[(t, key)] = alt_id
        if alt_id == base_item:
            continue
        b = base_result.catalog.get(base_item) if base_item else None
        bs, as_ = _status(index, b, key, t), _status(index, alt_pick, key, t)
        if kind == "audit":
            gained += bs != "correct" and as_ == "correct"
            lost += bs == "correct" and as_ != "correct"
        changes.append(
            ChangedDecision(
                kind=kind,
                at=t,
                key=key,
                base_item=base_item,
                alt_item=alt_id,
                base_value=b.value if b else None,
                alt_value=alt_pick.value if alt_pick else None,
                why=_explain(base_mem, alt_mem, b, alt_pick, key, t, field),
                base_status=bs,
                alt_status=as_,
            )
        )
    resim = simulate(world, alt, claims if alt.ingest == "extracted" else None)
    resim_changed = sum(
        x.item != y.item for x, y in zip(base_result.audits, resim.audits, strict=True)
    )
    diverged = sum(alt_audit[(x.t, x.key)] != x.item for x in resim.audits)
    if ledger_digest(base_result) != before or base_result.ledger.frozen != live_frozen:
        raise RuntimeError("the live run was modified by the counterfactual")
    reasons: dict[str, int] = defaultdict(int)
    for c in changes:
        reasons[c.why.kind] += 1
    return Counterfactual(
        world=world.digest,
        base=base.digest,
        alt=alt.digest,
        changed_field=field,
        evidence=before,
        decisions=len(points),
        changed=len(changes),
        changed_share=Proportion.of(len(changes), len(points), 0.95),
        by_why=tuple(sorted(reasons.items())),
        changes=tuple(changes),
        gained=gained,
        lost=lost,
        resimulated_changed=resim_changed,
        fixed_vs_resimulated=diverged,
        interpretation=INTERPRETATION,
    )


# --- substrate replay (Phases 1-7) ---------------------------------------------------------


def log_digest(log: MemoryLog) -> str:
    """Identity of a memory log (its canonical export). Replay must leave it unchanged."""
    return "sha256:" + hashlib.sha256(log.export()).hexdigest()


def substrate_state(log: MemoryLog, valid_at: datetime, known_at: datetime) -> MemoryState:
    """The bitemporal state as of ``known_at``: a pure fold, the log is not touched."""
    return log.state_as_of(valid_at=valid_at, known_at=known_at)


def substrate_hierarchy(
    log: MemoryLog, policy: ConsolidationPolicy, at: datetime, embedder: Embedder | None = None
) -> Hierarchy:
    """Consolidation as of ``at``, rebuilt from the log's history at that time (I50)."""
    return consolidate(Corpus.from_log(log, at), policy, at, embedder)
