"""The contradiction survey, the belief trace (autopsy prerequisites) and the end-to-end
demonstration of Phases 11-12."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import Field

from memoria.artifacts import ArtifactStore
from memoria.belief_lab import SELECTIVE, Summary, predictions, run_world, summarize
from memoria.belief_ops import memory_log
from memoria.belief_study import SUBGROUP_TAGS, reorder
from memoria.belief_worlds import CITIES, EMPLOYERS, BeliefWorld
from memoria.beliefs import POLICIES, BeliefState, check_arithmetic
from memoria.calibration import Isotonic, Prediction, by_subgroup
from memoria.consolidation import Hierarchy, consolidate
from memoria.contradictions import ClaimOntology, ContradictionGraph, contradiction_graph
from memoria.core import Digest, Record, UTCDatetime, quantize
from memoria.entities import CONSERVATIVE
from memoria.graph import GraphPolicy, GraphSnapshot, build_graph
from memoria.hybrid import Corpus
from memoria.memory_lab import HIERARCHICAL
from memoria.revision import (
    Decision,
    LedgerRun,
    RevisionEvent,
    Transition,
    covering,
    decide,
    historical_mismatches,
    reconstruct,
    replay,
    violations,
)
from memoria.scenarios import day
from memoria.statistics import Proportion

GRAPH_POLICY = GraphPolicy(name="belief-graph", version="1", resolution=CONSERVATIVE)


def world_ontology(window: float = 1.0) -> ClaimOntology:
    return ClaimOntology(
        name="worlds", version="1",
        categories=(("employer", tuple(sorted(e.lower() for e in EMPLOYERS))),
                    ("home", tuple(sorted(c.lower() for c in CITIES)))),
        concurrency_days=window,
    )  # fmt: skip


# --- contradiction survey ------------------------------------------------------------------------


class Survey(Record):
    world: str
    claims: int
    relations: tuple[tuple[str, int], ...]
    clusters: int
    open: int
    resolved: int
    genuine_precision: Proportion | None  # genuine relations with at least one erroneous report
    genuine_recall: Proportion | None  # erroneous reports concurrent with a correct one, flagged
    changes_misread: Proportion | None  # relations between two correct reports read as genuine
    graph: Digest
    contradictions: Digest


def survey(
    world: BeliefWorld, store: ArtifactStore | None = None, window: float = 1.0
) -> tuple[Survey, ContradictionGraph]:
    """Classify the contradictions of a world's claims and score them against the truth the
    generator kept. A genuine relation should involve an erroneous report (precision); an
    erroneous report made at the same time as a correct one should be flagged (recall); and a
    relation between two correct reports (a change over time) should not be genuine."""
    with memory_log(world) as log:
        end = day(world.spec.horizon_days + world.spec.delay_days * 2 + 30)
        corpus = Corpus.from_log(log, end)
        snap = build_graph(corpus, GRAPH_POLICY)
    cg = contradiction_graph(snap, world_ontology(window), world.sources)
    when = {n.id: n.occurred_at for n in snap.nodes if n.kind == "claim"}
    key_of = {c.id: c.key for c in snap.claims}
    wrong = {c.id: any(e in world.wrong_reports for e in c.evidence) for c in snap.claims}
    genuine = [c for c in cg.contradictions if c.genuine]
    flagged = {c.a for c in genuine if not wrong[c.b]} | {c.b for c in genuine if not wrong[c.a]}
    concurrent = [
        w for w in wrong if wrong[w] and any(
            not wrong[o] and key_of[o] == key_of[w] and o != w
            and abs((when[o] - when[w]).total_seconds()) <= 86400  # type: ignore[operator]
            for o in wrong
        )
    ]  # fmt: skip
    between_correct = [c for c in cg.contradictions if not wrong[c.a] and not wrong[c.b]]
    if store is not None:
        store.put_record(snap)
        store.put_record(cg)
    with_error = sum(1 for c in genuine if wrong[c.a] or wrong[c.b])
    result = Survey(
        world=world.spec.name, claims=len(wrong), relations=cg.counts, clusters=len(cg.clusters),
        open=sum(len(c.open) for c in cg.clusters),
        resolved=sum(len(c.resolved) for c in cg.clusters),
        genuine_precision=Proportion.of(with_error, len(genuine), 0.95) if genuine else None,
        genuine_recall=Proportion.of(
            sum(1 for w in concurrent if w in flagged), len(concurrent), 0.95
        ) if concurrent else None,
        changes_misread=Proportion.of(
            sum(1 for c in between_correct if c.genuine), len(between_correct), 0.95
        ) if between_correct else None,
        graph=snap.digest, contradictions=cg.digest,
    )  # fmt: skip
    return result, cg


# --- the belief trace ----------------------------------------------------------------------------


class TraceEvidence(Record):
    id: Digest
    role: str  # supporting | contradicting
    source: str
    root: str
    occurred_at: UTCDatetime
    recorded_at: UTCDatetime
    content: str | None  # the original experience, if the corpus has it
    memory_versions: tuple[Digest, ...]
    graph_claim: str | None
    derived_by: tuple[str, ...]  # consolidated memories covering it
    weight: float | None  # its contribution to the belief, if it contributed
    factors: tuple[tuple[str, float], ...]


class TraceEvent(Record):
    seq: int
    logical_time: UTCDatetime
    reason: str
    changes: tuple[str, ...]  # "belief: before -> after (score)"


class BeliefTrace(Record):
    """answer -> belief -> revision events -> confidence calculation -> evidence -> source ->
    claim -> memory version -> consolidation -> graph relations -> original experience.
    ``missing`` lists exactly which link could not be resolved (empty: complete)."""

    key: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    decision: Decision
    belief: str | None
    state: BeliefState | None
    score: float | None
    arithmetic: str  # "ok" or the violated identity
    weight: float | None
    mass: float | None
    prior_mass: float | None
    events: tuple[TraceEvent, ...]
    evidence: tuple[TraceEvidence, ...]
    graph_edges: tuple[str, ...]
    missing: tuple[str, ...]
    complete: bool = Field(default=False)


def _describe(t: Transition) -> str:
    before = t.before.value if t.before else "new"
    after = t.after.value if t.after else "revised"
    score = f" ({t.score_after:.3f})" if t.score_after is not None else ""
    return f"{t.belief}: {before} -> {after}{score}"


def belief_trace(
    run: LedgerRun, corpus: Corpus, hierarchy: Hierarchy | None, snapshot: GraphSnapshot | None,
    key: str, valid_at: datetime, known_at: datetime,
) -> BeliefTrace:  # fmt: skip
    kb = run.at(key, known_at)
    d = decide(key, kb, valid_at, known_at, SELECTIVE)
    chosen = next((b for b in covering(kb, valid_at) if b.id in d.beliefs), None)
    if chosen is None:
        return BeliefTrace(
            key=key, valid_at=valid_at, known_at=known_at, decision=d, belief=None, state=None,
            score=None, arithmetic="no belief", weight=None, mass=None, prior_mass=None,
            events=(), evidence=(), graph_edges=(), missing=("no belief answers this query",),
        )  # fmt: skip
    try:
        check_arithmetic(chosen)
        arithmetic = "ok"
    except ValueError as exc:
        arithmetic = str(exc)
    lineage = {chosen.id, *chosen.predecessors}
    last = kb.seq if kb else -1
    events = tuple(
        TraceEvent(
            seq=e.seq,
            logical_time=e.logical_time,
            reason=e.reason,
            changes=tuple(_describe(t) for t in e.transitions if t.belief in lineage),
        )
        for e in run.ledger.events
        if e.key == key and e.seq <= last and any(t.belief in lineage for t in e.transitions)
    )
    items = {i.id: i for i in run.ledger.evidence}
    contrib = {c.evidence: c for c in chosen.contributions if c.counted}
    entries = {e.digest: e for e in corpus.entries}
    graph_claim: dict[str, str] = {}
    edges: list[str] = []
    if snapshot is not None:
        for e in snapshot.edges:
            if e.relation == "asserts" and e.source.startswith("experience:"):
                graph_claim[e.source.removeprefix("experience:")] = e.target
    derived_by: dict[str, list[str]] = {}
    for m in hierarchy.memories if hierarchy else ():
        for x in m.derived_from:
            derived_by.setdefault(x, []).append(f"{m.level.value}:{m.memory_id}")
    missing: list[str] = []
    ev: list[TraceEvidence] = []
    for role, ids in (("supporting", chosen.supporting), ("contradicting", chosen.contradicting)):
        for x in ids:
            it = items[x]
            derived_item = bool(it.lineage)
            exp_ids = [e for e, _ in it.lineage] if derived_item else [x]
            exps = [corpus.experiences[e] for e in exp_ids if e in corpus.experiences]
            if len(exps) != len(exp_ids):
                missing.append(f"{x}: experience not in the corpus")
            versions = tuple(sorted(
                v.digest for v in entries.values()
                if v.level.value == "L1" and any(s.digest in exp_ids for s in v.sources)
            ))  # fmt: skip
            if not versions:
                missing.append(f"{x}: no memory version cites it")
            claim = graph_claim.get(exp_ids[0]) if exp_ids else None
            if snapshot is not None and claim is None and not derived_item:
                missing.append(f"{x}: no graph claim asserted by it")
            if claim is not None:
                edges.append(f"asserts|experience:{exp_ids[0]}|{claim}")
            c = contrib.get(x)
            ev.append(TraceEvidence(
                id=x, role=role, source=it.source, root=run.sources.root(it.source),
                occurred_at=it.occurred_at, recorded_at=it.recorded_at,
                content=exps[0].content if exps else None, memory_versions=versions,
                graph_claim=claim,
                derived_by=tuple(sorted({d for e in exp_ids for d in derived_by.get(e, [])})),
                weight=c.weight if c else None,
                factors=(("trust", c.trust), ("recency", c.recency), ("depth", c.depth),
                         ("staleness", c.staleness)) if c else (),
            ))  # fmt: skip
    if arithmetic != "ok":
        missing.append(f"confidence: {arithmetic}")
    return BeliefTrace(
        key=key, valid_at=valid_at, known_at=known_at, decision=d, belief=chosen.id,
        state=chosen.state, score=chosen.score, arithmetic=arithmetic, weight=chosen.weight,
        mass=chosen.mass, prior_mass=chosen.prior_mass, events=events, evidence=tuple(ev),
        graph_edges=tuple(sorted(set(edges))), missing=tuple(missing), complete=not missing,
    )  # fmt: skip


# --- the demonstration ---------------------------------------------------------------------------


class Trajectory(Record):
    key: str
    policy: str
    order: str
    points: tuple[tuple[float, str | None, float | None], ...]  # logical day, answer, score


class Demonstration(Record):
    """experiences -> memories -> consolidation -> graph -> claims -> contradictory evidence ->
    belief revision -> confidence -> selective prediction -> interference / forgetting ->
    recalibration -> provenance replay, on one world, with every claimed finding recorded."""

    world: str
    ledger: Digest
    ledger_h: Digest
    graph: Digest
    contradictions: Digest
    hierarchy: Digest
    # 1. legitimate temporal change is not a contradiction
    change: tuple[str, str, str, str]  # (key, earlier claim, later claim, earlier belief state)
    # 2. genuine contradiction stays explicit
    contradiction: tuple[str, str, str, int, int]  # (key, type, state, supporting, contradicting)
    # 3. a new evidence event changes belief state
    event: RevisionEvent
    # 4. order produces a measurable trajectory
    trajectories: tuple[Trajectory, Trajectory]
    trajectory_gap: float
    # 5. confidence evaluated against known outcomes
    summary: Summary
    # 6. a calibration failure found
    failure: str
    failure_detail: tuple[tuple[str, float], ...]
    # 7. abstention prevented an overconfident error
    abstention: tuple[str, str, str, float | None, str]  # (probe, policy, text, score, truth)
    # 8. provenance reaches the original experience
    trace: BeliefTrace
    replay_identical: bool
    violations: int
    historical_mismatches: int


def _find_change(cg: ContradictionGraph, snap: GraphSnapshot, run: LedgerRun,
                 world: BeliefWorld) -> tuple[str, str, str, str]:  # fmt: skip
    """A supersession between two correct reports whose earlier belief the ledger holds as
    SUPERSEDED: a change over time, kept as history, never a contradiction."""
    claims = {c.id: c for c in snap.claims}
    for c in cg.contradictions:
        a, b = claims[c.a], claims[c.b]
        clean = not any(e in world.wrong_reports for e in (*a.evidence, *b.evidence))
        if c.type == "supersession" and c.relation == "supersedes" and clean:
            kb = run.final(a.key)
            for x in kb.beliefs if kb else ():
                if x.value == a.object and x.state is BeliefState.SUPERSEDED:
                    return (a.key, a.id, b.id, x.state.value)
    return ("", "", "", "")


def _find_contradiction(cg: ContradictionGraph, snap: GraphSnapshot,
                        run: LedgerRun) -> tuple[str, str, str, int, int]:  # fmt: skip
    """A genuine (open) contradiction whose belief keeps evidence on both sides."""
    claims = {c.id: c for c in snap.claims}
    for c in cg.contradictions:
        if not c.genuine:
            continue
        key = claims[c.a].key
        kb = run.final(key)
        for b in kb.beliefs if kb else ():
            if b.contradicting and b.state is not BeliefState.SUPERSEDED:
                return (key, c.type, b.state.value, len(b.supporting), len(b.contradicting))
    return ("", "", "", 0, 0)


def _trajectory(run: LedgerRun, key: str, at: datetime, policy: str, order: str) -> Trajectory:
    points = []
    for kb in run.history[key]:
        cover = covering(kb, at)
        top = max(cover, key=lambda b: (b.score, b.id), default=None)
        days = round((kb.at - day(0)) / timedelta(days=1), 2)
        points.append((days, top.value if top else None, top.score if top else None))
    return Trajectory(key=key, policy=policy, order=order, points=tuple(points))


def _failure(
    preds: Sequence[Prediction], summary: Summary
) -> tuple[str, tuple[tuple[str, float], ...]]:
    detail: list[tuple[str, float]] = []
    parts: list[str] = []
    wrong = [p for p in preds if p.answered and p.correct is False and (p.score or 0) >= 0.6]
    if wrong:
        parts.append(f"{len(wrong)} answers with score >= 0.6 were wrong (first: {wrong[0].probe})")
        detail.append(("confident_wrong", float(len(wrong))))
    groups = by_subgroup(preds, SUBGROUP_TAGS)
    worst = max(((t, r) for t, r in groups.items() if r.n >= 8 and r.ece is not None),
                key=lambda x: x[1].ece or 0.0, default=None)  # fmt: skip
    if worst is not None:
        overall = summary.reliability.ece or 0.0
        parts.append(
            f"subgroup {worst[0]!r} has ECE {worst[1].ece:.2f} against {overall:.2f} overall"
        )
        detail += [
            ("worst_subgroup_ece", float(worst[1].ece or 0.0)),
            ("worst_subgroup_n", float(worst[1].n)),
        ]
    return "; ".join(parts) or "none observed", tuple(detail)


def _trajectory_gap(t1: Trajectory, t2: Trajectory) -> float:
    """Largest score difference between two trajectories on the union of their time grid
    (each read as the latest point at or before the time)."""

    def at(t: Trajectory, when: float) -> float:
        prior = [p for p in t.points if p[0] <= when]
        return (prior[-1][2] or 0.0) if prior else 0.0

    grid = sorted({p[0] for p in t1.points} | {p[0] for p in t2.points})
    return max((abs(at(t1, g) - at(t2, g)) for g in grid), default=0.0)


def demonstration(
    world: BeliefWorld, calibrators: dict[str, Isotonic], store: ArtifactStore | None = None,
    policy: str = "B:source-weighted", careful: str = "H:conservative",
    naive: str = "A:evidence-count",
) -> Demonstration:  # fmt: skip
    pol, pol_h = POLICIES[policy], POLICIES[careful]
    _, cg = survey(world, store)
    with memory_log(world) as log:
        end = day(world.spec.horizon_days + world.spec.delay_days * 2 + 30)
        corpus = Corpus.from_log(log, end)
        hier = consolidate(corpus, HIERARCHICAL, end)
        snap = build_graph(corpus.with_derived(hier.memories, ()), GRAPH_POLICY, [hier])
    run, run_h, run_a = (
        run_world(world, pol),
        run_world(world, pol_h),
        run_world(world, POLICIES[naive]),
    )
    change = _find_change(cg, snap, run, world)
    contradiction = _find_contradiction(cg, snap, run_h)
    moving = ("conflict_resolved", "correction_applied", "change_accepted")
    event = next(
        (e for r in moving for e in run.ledger.events if e.reason == r), run.ledger.events[0]
    )  # fmt: skip
    key = event.key
    at = day(max(s for s, _ in world.truth[key]) + 2)
    other = run_world(world, pol, evidence=reorder(world.evidence, 7, 20.0))
    t1 = _trajectory(run, key, at, policy, "as generated")
    t2 = _trajectory(other, key, at, policy, "reordered (seed 7, delay <= 20 days)")
    gap = _trajectory_gap(t1, t2)
    preds = predictions(run, world, calibrator=calibrators[policy])
    summary = summarize(preds)
    failure, detail = _failure(preds, summary)
    by_a = {p.probe: p for p in predictions(run_a, world)}
    by_h = {p.probe: p for p in predictions(run_h, world)}
    abstention: tuple[str, str, str, float | None, str] = ("", "", "", None, "")
    for pr in world.probes:
        a, h = by_a[pr.id], by_h[pr.id]
        if a.answered and a.correct is False and (a.score or 0) >= 0.5 and not h.answered:
            text = f"{naive} answered {a.value!r} (score {a.score:.2f}); {careful} chose {h.mode}"
            abstention = (pr.id, careful, text, a.score, pr.truth)
            break
    pr = next((p for p in world.probes if p.key == key), world.probes[0])
    trace = belief_trace(
        run, corpus, hier, snap, pr.key, day(pr.valid_at_day), day(pr.known_at_day)
    )
    identical, _ = replay(run.ledger, pol, world.sources)
    mid = day(world.spec.horizon_days / 2)
    hist = historical_mismatches(run, [mid, day(world.spec.horizon_days)])
    assert reconstruct(run.ledger, pol, world.sources, mid).ledger.events
    demo = Demonstration(
        world=world.spec.name, ledger=run.ledger.digest, ledger_h=run_h.ledger.digest,
        graph=snap.digest, contradictions=cg.digest, hierarchy=hier.digest, change=change,
        contradiction=contradiction, event=event, trajectories=(t1, t2),
        trajectory_gap=quantize(gap), summary=summary, failure=failure, failure_detail=detail,
        abstention=abstention, trace=trace, replay_identical=identical,
        violations=len(violations(run)), historical_mismatches=hist,
    )  # fmt: skip
    if store is not None:
        for rec in (run.ledger, run_h.ledger, cg, demo):
            store.put_record(rec)
    return demo
