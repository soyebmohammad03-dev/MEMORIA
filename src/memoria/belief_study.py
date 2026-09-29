"""The policy matrix, order sensitivity, stability, trust and uncertainty studies.

Worlds come in independent replicates (different seeds). Recalibration is fitted on the
calibration replicates (seed family 2) and evaluated on the test replicates (seed family 1).
Probes within a replicate share evidence, so the statistics have two levels: paired
per-probe tests (the design is paired: identical probes under two policies), and a
key-level analysis in which each (replicate, key) is one independent unit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta

from memoria.artifacts import ArtifactStore
from memoria.belief_lab import Summary, predictions, run_world, summarize
from memoria.belief_worlds import BeliefWorld, belief_world, worlds
from memoria.beliefs import POLICIES, BeliefPolicy, BeliefState, EvidenceItem
from memoria.calibration import (
    AbstentionQuality,
    Isotonic,
    Prediction,
    Reliability,
    abstention_quality,
    answered_pairs,
    auroc,
    by_subgroup,
    fit_isotonic,
    selective_accuracy_at,
)
from memoria.core import Digest, Record, quantize, unit_interval
from memoria.revision import LedgerRun, covering, historical_mismatches
from memoria.scenarios import day
from memoria.sources import spearman
from memoria.statistics import (
    PairedMean,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    paired_mean,
    significant,
)

WORLD_NAMES = tuple(s.name for s in worlds(1))
SUBGROUP_TAGS = (
    "contradiction-heavy",
    "temporal",
    "numeric",
    "entity-collision",
    "src:clinic",
    "src:email",
    "src:chat",
    "src:forum",
    "evidence:1",
    "evidence:2",
    "evidence:3+",
    "time:early",
    "time:mid",
    "time:late",
    "depth:1",
    "depth:2+",
)
BASELINE = "A:evidence-count"


class Runner:
    """Generates and caches worlds by (split, name, replicate)."""

    def __init__(self, replicates: int, cal_replicates: int) -> None:
        self.replicates = replicates
        self.cal_replicates = cal_replicates
        self._worlds: dict[tuple[str, str, int], BeliefWorld] = {}

    def world(self, split: str, name: str, r: int) -> BeliefWorld:
        k = (split, name, r)
        if k not in self._worlds:
            spec = next(s for s in worlds(1 if split == "test" else 2, r) if s.name == name)
            self._worlds[k] = belief_world(spec)
        return self._worlds[k]


# --- the policy matrix ---------------------------------------------------------------------------


class Comparison(Record):
    world: str
    policy: str
    baseline: str
    n: int
    correct: tuple[float | None, float | None, float | None, float | None, bool]
    keys: PairedMean  # key-level mean difference in correct answers (keys independent)
    brier: PairedMean | None  # key-level mean difference in squared error, jointly answered


class Cell(Record):
    world: str
    policy: str
    summary: Summary


class Subgroup(Record):
    policy: str
    tag: str
    raw: Reliability
    calibrated: Reliability | None


class MatrixResult(Record):
    cells: tuple[Cell, ...]
    pooled: tuple[Cell, ...]  # per policy over all worlds
    comparisons: tuple[Comparison, ...]
    calibrators: tuple[tuple[str, Isotonic], ...]
    subgroups: tuple[Subgroup, ...]
    quality: tuple[tuple[str, AbstentionQuality], ...]
    uncertainty: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...]
    ledgers: tuple[tuple[str, str, Digest], ...]  # (world, policy, ledger) test replicate 0


def _vec(preds: Sequence[Prediction]) -> list[bool]:
    return [bool(p.answered and p.correct) for p in preds]


def _cluster(
    base: list[Prediction], treat: list[Prediction], keys: list[str], conf: float
) -> PairedMean:
    per: dict[str, list[float]] = {}
    for a, b, k in zip(base, treat, keys, strict=True):
        per.setdefault(k, []).append(
            float(bool(b.answered and b.correct)) - float(bool(a.answered and a.correct))
        )
    return paired_mean([math.fsum(v) / len(v) for _, v in sorted(per.items())], conf)


def _brier_keys(
    base: list[Prediction], treat: list[Prediction], keys: list[str], conf: float
) -> PairedMean | None:
    per: dict[str, list[float]] = {}
    for a, b, k in zip(base, treat, keys, strict=True):
        if a.answered and b.answered and a.score is not None and b.score is not None:
            y = float(bool(a.correct)) if a.value == b.value else None
            if y is None:
                continue
            per.setdefault(k, []).append((b.score - y) ** 2 - (a.score - y) ** 2)
    if not per:
        return None
    return paired_mean([math.fsum(v) / len(v) for _, v in sorted(per.items())], conf)


def matrix(
    runner: Runner,
    store: ArtifactStore | None,
    policies: Sequence[str] | None = None,
    confidence: float = 0.95,
    world_names: Sequence[str] = WORLD_NAMES,
) -> MatrixResult:
    names = list(policies or sorted(POLICIES))
    # 1. Fit recalibration on the calibration replicates (raw memory, all worlds pooled).
    calibrators: dict[str, Isotonic] = {}
    for pname in names:
        pairs = []
        for wn in world_names:
            for r in range(runner.cal_replicates):
                w = runner.world("cal", wn, r)
                pairs += answered_pairs(predictions(run_world(w, POLICIES[pname]), w))
        calibrators[pname] = fit_isotonic(pairs)
    # 2. Evaluate on the test replicates.
    preds: dict[tuple[str, str], list[Prediction]] = {}
    keys: dict[str, list[str]] = {}
    ledgers: list[tuple[str, str, Digest]] = []
    for wn in world_names:
        for r in range(runner.replicates):
            w = runner.world("test", wn, r)
            keys.setdefault(wn, []).extend(f"{r}:{p.key}" for p in w.probes)
            for pname in names:
                run = run_world(w, POLICIES[pname])
                preds.setdefault((wn, pname), []).extend(
                    predictions(run, w, calibrator=calibrators[pname])
                )
                if r == 0 and store is not None:
                    ledgers.append((wn, pname, store.put_record(run.ledger)))
    cells = tuple(
        Cell(world=wn, policy=p, summary=summarize(preds[(wn, p)], confidence))
        for wn in world_names
        for p in names
    )
    pooled_preds = {p: [x for wn in world_names for x in preds[(wn, p)]] for p in names}
    pooled = tuple(
        Cell(world="ALL", policy=p, summary=summarize(pooled_preds[p], confidence)) for p in names
    )
    # 3. Paired comparisons against the baseline policy, Holm within world.
    comparisons: list[Comparison] = []
    for wn in [*world_names, "ALL"]:
        base = pooled_preds[BASELINE] if wn == "ALL" else preds[(wn, BASELINE)]
        ks = [k for w in world_names for k in keys[w]] if wn == "ALL" else keys[wn]
        if wn == "ALL":
            ks = [f"{w}/{k}" for w in world_names for k in keys[w]]
        rows = []
        for p in names:
            if p == BASELINE:
                continue
            treat = pooled_preds[p] if wn == "ALL" else preds[(wn, p)]
            a, b = _vec(base), _vec(treat)
            both = sum(x and y for x, y in zip(a, b, strict=True))
            f = sum(x and not y for x, y in zip(a, b, strict=True))  # baseline only
            g = sum(y and not x for x, y in zip(a, b, strict=True))  # treatment only
            diff = newcombe_paired(both, g, f, len(a) - both - f - g, confidence)
            rows.append(
                (
                    p,
                    len(a),
                    diff,
                    significant(mcnemar_exact(f, g)),
                    min_achievable_p(f + g) > 0.05,
                    treat,
                )
            )
        adjusted = holm([r[3] for r in rows])
        for (p, n, diff, _, under, treat), q in zip(rows, adjusted, strict=True):
            comparisons.append(
                Comparison(
                    world=wn,
                    policy=p,
                    baseline=BASELINE,
                    n=n,
                    correct=(
                        quantize(diff[0]),
                        quantize(diff[1]),
                        quantize(diff[2]),
                        significant(q),
                        under,
                    )
                    if diff
                    else (None, None, None, None, True),
                    keys=_cluster(base, treat, ks, confidence),
                    brier=_brier_keys(base, treat, ks, confidence),
                )
            )
    subgroups = []
    for p in names:
        raw = by_subgroup(pooled_preds[p], SUBGROUP_TAGS)
        cal = by_subgroup(pooled_preds[p], SUBGROUP_TAGS, calibrated=True)
        for t in SUBGROUP_TAGS:
            subgroups.append(
                Subgroup(policy=p, tag=t, raw=raw[t], calibrated=cal[t] if cal[t].n else None)
            )
    return MatrixResult(
        cells=cells,
        pooled=pooled,
        comparisons=tuple(comparisons),
        calibrators=tuple(sorted(calibrators.items())),
        subgroups=tuple(subgroups),
        quality=tuple((p, abstention_quality(pooled_preds[p])) for p in names),
        uncertainty=tuple((p, uncertainty_auroc(pooled_preds[p])) for p in names),
        ledgers=tuple(ledgers),
    )


def uncertainty_auroc(preds: Sequence[Prediction]) -> tuple[tuple[str, float | None], ...]:
    """Does each uncertainty component rank errors above correct answers? (0.5: no.)
    The confidence score's complement is included for comparison."""
    answered = [p for p in preds if p.answered and p.correct is not None and p.uncertainty]
    out: list[tuple[str, float | None]] = []
    names = [n for n, _ in answered[0].uncertainty] if answered else []
    for n in names:
        out.append((n, auroc([(dict(p.uncertainty)[n], not p.correct) for p in answered])))
    out.append(("1-score", auroc([(1 - (p.score or 0.0), not p.correct) for p in answered])))
    return tuple(out)


# --- stability -----------------------------------------------------------------------------------


class Stability(Record):
    world: str
    policy: str
    keys: int
    events: int
    revisions_per_event: float  # belief state changes per event
    churn: float  # answer changes per key (at the key's current valid time)
    oscillations: float  # returns to a previously held answer, per key
    confidence_volatility: float | None  # mean |score change| of the answering belief
    persistence: float | None  # share of post-conflict states still in conflict
    time_to_resolution_days: float | None  # mean logical days of a resolved conflict episode
    unresolved_at_end: int
    latency_days: float | None  # occurrence-to-revision delay of belief-changing evidence


def stability(run: LedgerRun, world: BeliefWorld, policy: str) -> Stability:
    flips = osc = 0
    vol: list[float] = []
    in_conflict = post = 0
    episodes: list[float] = []
    open_end = 0
    for key, states in run.history.items():
        t = day(max(s for s, _ in world.truth[key]) + 2)
        held: list[str] = []
        prev_score: float | None = None
        started = None
        for kb in states:
            cover = covering(kb, t)
            single = cover[0] if len(cover) == 1 else None
            conflict = any(
                b.state in (BeliefState.CONTESTED, BeliefState.UNRESOLVED) for b in cover
            )
            if single is not None and single.state is BeliefState.SUPPORTED:
                if held and held[-1] != single.value:
                    flips += 1
                    osc += single.value in held[:-1]
                if not held or held[-1] != single.value:
                    held.append(single.value)
            score = max((b.score for b in cover), default=None)
            if score is not None and prev_score is not None:
                vol.append(abs(score - prev_score))
            prev_score = score if score is not None else prev_score
            if conflict:
                started = started or kb.at
            elif started is not None:
                episodes.append((kb.at - started) / timedelta(days=1))
                started = None
            if started is not None or episodes:
                post += 1
                in_conflict += conflict
        open_end += started is not None
    moved = [
        e
        for e in run.ledger.events
        if e.reason in ("change_accepted", "conflict_resolved", "correction_applied")
    ]
    occurred = {i.id: i.occurred_at for i in run.ledger.evidence}
    delays = [(e.logical_time - occurred[e.evidence]) / timedelta(days=1) for e in moved]
    changes = sum(1 for e in run.ledger.events for t in e.transitions if t.before != t.after)
    nk = max(1, len(run.history))
    return Stability(
        world=world.spec.name,
        policy=policy,
        keys=len(run.history),
        events=len(run.ledger.events),
        revisions_per_event=quantize(changes / max(1, len(run.ledger.events))),
        churn=quantize(flips / nk),
        oscillations=quantize(osc / nk),
        confidence_volatility=quantize(math.fsum(vol) / len(vol)) if vol else None,
        persistence=quantize(in_conflict / post) if post else None,
        time_to_resolution_days=quantize(math.fsum(episodes) / len(episodes)) if episodes else None,
        unresolved_at_end=open_end,
        latency_days=quantize(math.fsum(delays) / len(delays)) if delays else None,
    )


# --- order sensitivity ---------------------------------------------------------------------------


def reorder(evidence: Sequence[EvidenceItem], seed: int, max_delay: float) -> list[EvidenceItem]:
    """The same evidence, arriving in another order: arrival = occurrence + a seeded delay."""
    return [
        e.model_copy(
            update={
                "recorded_at": e.occurred_at
                + timedelta(days=unit_interval(seed, "arrival", e.id) * max_delay)
            }
        )
        for e in evidence
    ]


class OrderResult(Record):
    world: str
    policy: str
    orders: int
    same_final: int  # orders whose final beliefs equal the reference order's
    answer_agreement: float  # mean share of probes with the reference order's final answer
    mid_agreement: float  # the same at a mid-horizon checkpoint (intermediate belief)
    revisions: tuple[int, float, int]  # min, mean, max state changes
    correct: tuple[float, float, float]  # min, mean, max share of probes answered correctly
    confidence_range: float  # mean over probes of (max - min) answering score across orders
    historical_mismatches: int  # reconstructions differing from the live run, over checked orders


def _finals(run: LedgerRun) -> dict[str, str]:
    return {k: kb.content_fingerprint() for k in run.history if (kb := run.final(k)) is not None}


def _final_answers(
    run: LedgerRun, world: BeliefWorld, horizon: float
) -> dict[str, tuple[str | None, float | None]]:
    out: dict[str, tuple[str | None, float | None]] = {}
    for p in world.probes:
        kb = run.at(p.key, day(horizon))
        cover = covering(kb, day(p.valid_at_day))
        one = (
            cover[0]
            if len(cover) == 1 and cover[0].state in (BeliefState.SUPPORTED, BeliefState.SUPERSEDED)
            else None
        )
        out[p.id] = (one.value if one else None, one.score if one else None)
    return out


def order_study(
    world: BeliefWorld, policy: BeliefPolicy, orders: int, max_delay: float = 20.0, checked: int = 2
) -> OrderResult:
    horizon = world.spec.horizon_days + max_delay + 5
    mid = world.spec.horizon_days / 2
    runs = [
        run_world(world, policy, evidence=reorder(world.evidence, o, max_delay))
        for o in range(orders)
    ]
    ref = runs[0]
    ref_final = _finals(ref)
    ref_ans = _final_answers(ref, world, horizon)
    ref_mid = _final_answers(ref, world, mid)
    same = agree = mid_agree = 0.0
    revs: list[int] = []
    corr: list[float] = []
    ranges: dict[str, list[float]] = {}
    for r in runs:
        same += _finals(r) == ref_final
        ans, md = _final_answers(r, world, horizon), _final_answers(r, world, mid)
        agree += sum(ans[p][0] == ref_ans[p][0] for p in ans) / len(ans)
        mid_agree += sum(md[p][0] == ref_mid[p][0] for p in md) / len(md)
        revs.append(sum(1 for e in r.ledger.events for t in e.transitions if t.before != t.after))
        truth = {p.id: p.truth for p in world.probes}
        corr.append(sum(ans[p][0] == truth[p] for p in ans) / len(ans))
        for p, (_, sc) in ans.items():
            if sc is not None:
                ranges.setdefault(p, []).append(sc)
    times = [day(mid), day(world.spec.horizon_days)]
    bad = sum(historical_mismatches(r, times) for r in runs[:checked])
    spread = [max(v) - min(v) for v in ranges.values() if len(v) > 1]
    return OrderResult(
        world=world.spec.name,
        policy=policy.name,
        orders=orders,
        same_final=int(same),
        answer_agreement=quantize(agree / orders),
        mid_agreement=quantize(mid_agree / orders),
        revisions=(min(revs), quantize(sum(revs) / orders), max(revs)),
        correct=(quantize(min(corr)), quantize(sum(corr) / orders), quantize(max(corr))),
        confidence_range=quantize(math.fsum(spread) / len(spread)) if spread else 0.0,
        historical_mismatches=bad,
    )


# --- trust ---------------------------------------------------------------------------------------


class TrustResult(Record):
    world: str
    sources: int
    mae: float | None
    spearman: float | None
    circular: bool
    best_learned: tuple[str, float] | None
    worst_learned: tuple[str, float] | None


def trust_study(world: BeliefWorld, run: LedgerRun) -> TrustResult:
    assert run.trust is not None
    recs = [
        r
        for r in run.trust.snapshot()
        if r.reports >= 3 and r.source in world.reliability and not r.source.startswith("bot:")
    ]
    learned = {r.source: r.mean for r in recs}
    truth = {s: world.reliability[s] for s in learned}
    mae = (
        quantize(sum(abs(learned[s] - truth[s]) for s in learned) / len(learned))
        if learned
        else None
    )
    ranked = sorted(learned.items(), key=lambda x: (-x[1], x[0]))
    return TrustResult(
        world=world.spec.name,
        sources=len(learned),
        mae=mae,
        spearman=spearman(
            [learned[s] for s in sorted(learned)], [truth[s] for s in sorted(learned)]
        ),
        circular=any(r.circular for r in recs),
        best_learned=ranked[0] if ranked else None,
        worst_learned=ranked[-1] if ranked else None,
    )


class SelectiveResult(Record):
    policy: str
    aurc: float | None
    oracle_aurc: float | None
    sel_at_50: float | None
    sel_at_80: float | None
    accuracy: float | None
    coverage: float


def selective_study(pooled: Sequence[tuple[str, Summary]]) -> tuple[SelectiveResult, ...]:
    return tuple(
        SelectiveResult(
            policy=p, aurc=s.risk.aurc, oracle_aurc=s.risk.oracle_aurc,
            sel_at_50=selective_accuracy_at(s.risk, 0.5),
            sel_at_80=selective_accuracy_at(s.risk, 0.8), accuracy=s.risk.accuracy,
            coverage=quantize(s.answered / max(1, s.n)),
        )
        for p, s in pooled
    )  # fmt: skip
