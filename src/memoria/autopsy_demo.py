"""A fixed, deterministic demonstration of the autopsy, replay and counterfactual layer.

Everything is chosen by rule from one generated world (drifting, replicate 0 of the benchmark's
seed family) and one belief world, so the demonstration is a pure function of the code and is
reproduced into an independent store like any other artifact.
"""

from __future__ import annotations

from memoria.adaptive import ablate
from memoria.adaptive_worlds import build_world
from memoria.artifacts import ArtifactStore
from memoria.autopsy import (
    Autopsy,
    ChangedDecision,
    Counterfactual,
    StateSnapshot,
    belief_autopsy,
    belief_context,
    counterfactual,
    memory_autopsy,
    snapshot,
)
from memoria.belief_demo import belief_trace
from memoria.belief_lab import predictions, run_world
from memoria.belief_worlds import belief_world
from memoria.beliefs import POLICIES
from memoria.benchmark import (
    BELIEF_TEST,
    ENVIRONMENTS,
    MEMORY_TEST,
    BeliefContext,
    belief_spec,
    bench_spec,
)
from memoria.core import Digest, Record
from memoria.retention import KEEP, System, gate, precompute_claims, simulate
from memoria.scenarios import day
from memoria.statistics import Proportion


class CounterfactualSummary(Record):
    label: str
    changed_field: str
    decisions: int
    changed: int
    changed_share: Proportion
    by_why: tuple[tuple[str, int], ...]
    gained: int
    lost: int
    resimulated_changed: int
    fixed_vs_resimulated: int
    example: ChangedDecision | None
    evidence: Digest
    interpretation: str


class ReplayPair(Record):
    """The same key replayed just before and just after a source correction was recorded."""

    key: str
    item: str
    event: str
    recorded_at: float
    before: StateSnapshot
    after: StateSnapshot
    item_before: tuple[str, str]
    item_after: tuple[str, str]


class AutopsyDemonstration(Record):
    world: Digest
    memory_autopsy: Autopsy
    replay: ReplayPair | None
    counterfactuals: tuple[CounterfactualSummary, ...]
    belief_autopsy: Autopsy


def _summary(label: str, cf: Counterfactual) -> CounterfactualSummary:
    return CounterfactualSummary(
        label=label,
        changed_field=cf.changed_field,
        decisions=cf.decisions,
        changed=cf.changed,
        changed_share=cf.changed_share,
        by_why=cf.by_why,
        gained=cf.gained,
        lost=cf.lost,
        resimulated_changed=cf.resimulated_changed,
        fixed_vs_resimulated=cf.fixed_vs_resimulated,
        example=cf.changes[0] if cf.changes else None,
        evidence=cf.evidence,
        interpretation=cf.interpretation,
    )


def demonstrate(store: ArtifactStore | None = None) -> AutopsyDemonstration:
    _, mem, bel = ENVIRONMENTS[1]  # drifting
    world = build_world(bench_spec(MEMORY_TEST, 0, mem))
    claims = precompute_claims(world)
    f = System(name="F:gate-adaptive", schedule=gate(0.5), rank_weight=0.5)
    g = System(name="G:gate-stale-aware", schedule=gate(0.5, stale_aware=True), rank_weight=0.5)
    b = System(name="B:adaptive-rank", schedule=KEEP, rank_weight=0.5)
    res_f = simulate(world, f, claims)
    # the first late audit answered with a value that is no longer true and has a correction or
    # contradiction on record
    events = {(e.item, e.kind) for e in (res_f.ledger.all() if res_f.ledger else [])}
    pick = next(
        (
            a
            for a in res_f.audits
            if (a.t > 100 and a.status == "stale" and a.item and (a.item, "correction") in events)
            or (a.item, "contradiction_discovered") in events
        ),
        next(a for a in res_f.audits if a.item),
    )
    au = memory_autopsy(world, f, res_f, claims, pick.key, pick.t, "audit")
    # replay around a source correction, stale-aware system
    res_g = simulate(world, g, claims)
    pair: ReplayPair | None = None
    assert res_g.ledger is not None
    for e in res_g.ledger.all():
        if e.kind != "correction" or e.origin != "source_claim":
            continue
        it = res_g.catalog[e.item]
        keys = [it.key]
        s0 = snapshot(res_g, g, e.recorded_at, keys)
        s1 = snapshot(res_g, g, e.recorded_at + 0.001, keys)
        st0 = next((s, r) for i, s, r in s0.states if i == it.id)
        st1 = next((s, r) for i, s, r in s1.states if i == it.id)
        if st0 != st1:
            pair = ReplayPair(
                key=it.key,
                item=it.id,
                event=e.id,
                recorded_at=e.recorded_at,
                before=s0,
                after=s1,
                item_before=st0,
                item_after=st1,
            )
            break
    keep_b = simulate(world, b, claims)
    f_up = f.model_copy(update={"rank_weight": 1.0})
    f_minus = f.model_copy(update={"importance": ablate(f.importance, "correction")})
    cfs = (
        _summary("F: rank weight 0.5 -> 1.0", counterfactual(world, f, f_up, res_f, claims)),
        _summary("B -> F: keep all -> gate 0.5", counterfactual(world, b, f, keep_b, claims)),
        _summary("F: remove correction", counterfactual(world, f, f_minus, res_f, claims)),
    )
    # a belief answer
    bw = belief_world(belief_spec(BELIEF_TEST, 0, bel))
    ctx = BeliefContext(bw)
    run = run_world(bw, POLICIES["B:source-weighted"])
    preds = predictions(run, bw)
    ans = next(p for p in preds if p.answered and p.correct)
    pr = next(p for p in bw.probes if p.id == ans.probe)
    known = day(pr.known_at_day)
    trace = belief_trace(run, ctx.corpus, ctx.hier, ctx.snap, pr.key, day(pr.valid_at_day), known)
    cands, later = belief_context(run, pr.key, known)
    bau = belief_autopsy(
        bw.dataset.digest,
        "B:source-weighted",
        trace,
        later,
        pr.key,
        pr.known_at_day,
        ans.value,
        cands,
    )
    demo = AutopsyDemonstration(
        world=world.digest, memory_autopsy=au, replay=pair, counterfactuals=cfs, belief_autopsy=bau
    )
    if store is not None:
        store.put_record(demo)
    return demo
