"""The Super-Phase 5 study: adaptive importance, retention dynamics, extraction and identity.

Design. Five generated worlds (stable, changing, repetitive, noisy, adversarial) x declared
memory systems x independent replicates (a replicate is a differently seeded world, so
replicates are independent units and the pairing is across systems within a replicate).
Every system runs the same chronological simulation (:mod:`memoria.retention`); audit probes
measure each key every ten days without creating events.

Statistics. Proportions carry Wilson intervals over pooled probes (probes of one key are
correlated: a caveat, not corrected). *Inference uses replicates as the unit*: per-replicate
rates, paired differences against a named baseline, a Student-t interval, an exact sign test,
Holm within a family (one world x one metric x the listed comparisons), and an underpowered
flag (:func:`memoria.statistics.paired_mean`). A change is called degraded/improved only when
the adjusted p < alpha, the sign test is not underpowered and the mean has that sign. Nothing
here is a universal claim: worlds are generated, weights, thresholds and doses are declared and
untuned, and effects are observed in this design only.

Onset. For a treatment against the baseline, the *onset of damage* is the earliest checkpoint
where the paired stale-rate difference is degraded (Holm across checkpoints); for a dose it is
the smallest dose so degraded (Holm across doses). Resolution is the checkpoint or dose grid.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field

from memoria.adaptive import (
    COMPONENTS,
    DEFAULT,
    STATIC,
    ablate,
    verify_importance,
)
from memoria.adaptive_cases import (
    ExtractionStats,
    IdentityStats,
    extraction_stats,
    identity_stats,
    merge_cases,
    pool_extraction,
    pool_identity,
    run_extraction_cases,
    run_identity_cases,
)
from memoria.adaptive_worlds import (
    CHECKPOINTS,
    WORLD_NAMES,
    World,
    build_world,
    world_specs,
)
from memoria.artifacts import ArtifactStore
from memoria.calibration import Isotonic, auroc, fit_isotonic, reliability
from memoria.claims import Claim
from memoria.core import Digest, Record, quantize
from memoria.identity import CONSERVATIVE, SIMILARITY, IdentityPolicy
from memoria.retention import (
    KEEP,
    WARMUP_DAYS,
    AuditProbe,
    SimResult,
    System,
    age_window,
    gate,
    precompute_claims,
    simulate,
)
from memoria.semantic_eval import PerformanceRecord, environment
from memoria.statistics import PairedMean, Proportion, holm, paired_mean, significant

CONFIDENCE = 0.95
ALPHA = 0.05
WINDOW = 30.0  # a checkpoint T summarises audit probes in (T - WINDOW, T]
DOSE_LAMBDAS = (0.25, 0.5, 0.75, 1.0)
DOSE_THETAS = (0.45, 0.5, 0.55, 0.6, 0.65)


# --- systems ------------------------------------------------------------------------------


def systems() -> dict[str, System]:
    """Every system of the study by name. A is the static baseline; declared, untuned values."""
    out: dict[str, System] = {}

    def add(s: System) -> None:
        out[s.name] = s

    add(System(name="A:static-latest", schedule=KEEP))
    add(System(name="B:adaptive-rank", schedule=KEEP, rank_weight=0.5))
    add(System(name="C:age-30", schedule=age_window(30)))
    add(System(name="D:age-90", schedule=age_window(90)))
    add(System(name="E:gate-latest", schedule=gate(0.5)))
    add(System(name="F:gate-adaptive", schedule=gate(0.5), rank_weight=0.5))
    add(System(name="G:gate-stale-aware", schedule=gate(0.5, stale_aware=True), rank_weight=0.5))
    add(System(name="H:static-importance", schedule=KEEP, rank_weight=0.5, importance=STATIC))
    add(System(name="I:gold-extraction", schedule=KEEP, ingest="gold"))
    add(System(name="J:similarity-identity", schedule=KEEP, identity="similarity"))
    add(System(name="K:feedback-lag-7", schedule=gate(0.5), rank_weight=0.5, feedback_lag_days=7))
    for lam in DOSE_LAMBDAS:
        if lam != 0.5:
            add(System(name=f"R:rank-{lam:g}", schedule=KEEP, rank_weight=lam))
    for th in DOSE_THETAS:
        if th != 0.5:
            add(System(name=f"T:gate-{th:g}", schedule=gate(th), rank_weight=0.5))
    for c in COMPONENTS:
        add(
            System(
                name=f"X:minus-{c}",
                schedule=gate(0.5),
                rank_weight=0.5,
                importance=ablate(DEFAULT, c),
            )
        )
    return out


BASELINE = "A:static-latest"
ADAPTIVE = "F:gate-adaptive"
# (treatment, baseline): the comparisons of each world's family
COMPARISONS: tuple[tuple[str, str], ...] = (
    *(
        (s, BASELINE)
        for s in (
            "B:adaptive-rank",
            "C:age-30",
            "D:age-90",
            "E:gate-latest",
            "F:gate-adaptive",
            "G:gate-stale-aware",
            "H:static-importance",
            "I:gold-extraction",
            "J:similarity-identity",
            "R:rank-0.25",
            "R:rank-0.75",
            "R:rank-1",
            "T:gate-0.45",
            "T:gate-0.55",
            "T:gate-0.6",
            "T:gate-0.65",
        )
    ),
    ("F:gate-adaptive", "E:gate-latest"),  # ranking by importance under the gate
    ("G:gate-stale-aware", "F:gate-adaptive"),  # supersession-aware retention
    ("F:gate-adaptive", "H:static-importance"),  # adaptive against event-free importance
    ("K:feedback-lag-7", "F:gate-adaptive"),  # late-arriving feedback
    ("F:gate-adaptive", "B:adaptive-rank"),  # the gate alone, ranking held fixed
    ("G:gate-stale-aware", "B:adaptive-rank"),  # stale-aware gate alone, ranking held fixed
)
ABLATIONS = tuple(f"X:minus-{c}" for c in COMPONENTS)
RANK_DOSE = ("A:static-latest", "R:rank-0.25", "B:adaptive-rank", "R:rank-0.75", "R:rank-1")
GATE_DOSE = ("T:gate-0.45", "F:gate-adaptive", "T:gate-0.55", "T:gate-0.6", "T:gate-0.65")
ONSETS: tuple[tuple[str, str], ...] = (
    ("B:adaptive-rank", "A:static-latest"),
    ("C:age-30", "A:static-latest"),
    ("E:gate-latest", "A:static-latest"),
    ("F:gate-adaptive", "A:static-latest"),
    ("G:gate-stale-aware", "A:static-latest"),
    ("R:rank-1", "A:static-latest"),
    ("F:gate-adaptive", "B:adaptive-rank"),
    ("G:gate-stale-aware", "B:adaptive-rank"),
    ("T:gate-0.65", "B:adaptive-rank"),
)
CAL_SYSTEMS = ("A:static-latest", "B:adaptive-rank")


# --- per-run records ----------------------------------------------------------------------


class Counts(Record):
    """Audit-probe counts over a time window (all probes traceable to AuditProbe rows)."""

    n: int
    correct: int
    stale: int
    wrong: int
    none: int
    answered: int
    lineage_ok: int
    evidence_known: int  # probes where some recorded item supports the current truth
    evidence_kept: int  # ... and a retrievable item does
    available: int  # retrievable items summed over probes
    known: int  # recorded items summed over probes


class Replicate(Record):
    """One system on one world replicate."""

    world: str
    system: str
    replicate: int
    seed: int
    world_digest: Digest
    total: Counts  # probes after the warm-up
    checkpoints: tuple[tuple[float, Counts], ...]
    latency_changes: int
    latency_days: float  # restricted: censored at the end of the period
    latency_censored: int
    items: int
    events: int
    scanned: int
    importance_evals: int
    text_bytes: int
    claims: int
    misattributed: int
    unresolved: tuple[tuple[str, int], ...]
    user: tuple[tuple[str, int], ...]  # the simulated users' outcomes: correct, wrong, none


def counts(audits: Sequence[AuditProbe], lo: float, hi: float) -> Counts:
    sel = [a for a in audits if lo < a.t <= hi]
    return Counts(
        n=len(sel),
        correct=sum(a.status == "correct" for a in sel),
        stale=sum(a.status == "stale" for a in sel),
        wrong=sum(a.status == "wrong" for a in sel),
        none=sum(a.status == "none" for a in sel),
        answered=sum(a.status != "none" for a in sel),
        lineage_ok=sum(a.lineage_ok is True for a in sel),
        evidence_known=sum(a.evidence_known for a in sel),
        evidence_kept=sum(a.evidence_known and a.evidence_available for a in sel),
        available=sum(a.available for a in sel),
        known=sum(a.known for a in sel),
    )


def replicate_record(world: World, system: System, replicate: int, res: SimResult) -> Replicate:
    lat = res.latency
    return Replicate(
        world=world.name,
        system=system.name,
        replicate=replicate,
        seed=world.seed,
        world_digest=world.digest,
        total=counts(res.audits, WARMUP_DAYS, 1e9),
        checkpoints=tuple((h, counts(res.audits, h - WINDOW, h)) for h in CHECKPOINTS),
        latency_changes=len(lat),
        latency_days=quantize(math.fsum(d for _, d, _ in lat)),
        latency_censored=sum(c for _, _, c in lat),
        items=res.items,
        events=res.events,
        scanned=res.scanned,
        importance_evals=res.importance_evals,
        text_bytes=res.text_bytes,
        claims=res.claims,
        misattributed=res.misattributed,
        unresolved=tuple(sorted(res.unresolved.items())),
        user=tuple(sorted(res.user_status.items())),
    )


# --- statistics helpers -------------------------------------------------------------------


class Interval(Record):
    """Mean of per-replicate values with a Student-t interval (independent replicates)."""

    n: int
    mean: float | None
    low: float | None
    high: float | None


def interval(values: Sequence[float]) -> Interval:
    pm = paired_mean(list(values), CONFIDENCE, ALPHA)
    return Interval(n=pm.n, mean=pm.mean, low=pm.low, high=pm.high)


MetricFn = Callable[[Counts], tuple[int, int]]
METRICS: dict[str, MetricFn] = {
    "stale_rate": lambda c: (c.stale, c.n),
    "correct_rate": lambda c: (c.correct, c.n),
    "wrong_rate": lambda c: (c.wrong, c.n),
    "none_rate": lambda c: (c.none, c.n),
    "evidence_retention": lambda c: (c.evidence_kept, c.evidence_known),
    "item_retention": lambda c: (c.available, c.known),
    "lineage_complete": lambda c: (c.lineage_ok, c.answered),
}
CURVE_METRICS = ("stale_rate", "correct_rate", "evidence_retention", "item_retention")


def rep_values(
    reps: Sequence[Replicate], metric: str, window: float | None = None
) -> list[float | None]:
    """Per-replicate rate (None where the denominator is 0), in replicate order."""
    out: list[float | None] = []
    for r in reps:
        c = r.total if window is None else dict(r.checkpoints)[window]
        k, n = METRICS[metric](c)
        out.append(k / n if n else None)
    return out


def pooled(
    reps: Sequence[Replicate], metric: str, window: float | None = None
) -> Proportion | None:
    ks = ns = 0
    for r in reps:
        c = r.total if window is None else dict(r.checkpoints)[window]
        k, n = METRICS[metric](c)
        ks, ns = ks + k, ns + n
    return Proportion.of(ks, ns, CONFIDENCE) if ns else None


def paired(a: Sequence[float | None], b: Sequence[float | None]) -> PairedMean:
    """Mean of (a - b) over replicates where both are defined."""
    return paired_mean(
        [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None],
        CONFIDENCE,
        ALPHA,
    )


class Summary(Record):
    """Full-horizon measurements of one system in one world."""

    world: str
    system: str
    replicates: int
    metrics: tuple[tuple[str, Proportion | None, Interval], ...]  # pooled and across replicates
    latency_days: Interval  # mean restricted revision latency per replicate (days)
    latency_censored: Proportion | None  # changes never reflected in an answer before the next
    probes: int
    items_per_probe_available: float | None  # memory footprint: retrievable items per probe
    scanned_per_probe: float | None  # items considered per decision (latency proxy)
    events: int
    importance_evals: int
    misattributed: Proportion | None  # extracted claims filed under the wrong entity
    unresolved: Proportion | None  # unresolved claim-like sentences over all claim-like sentences
    user_correct: Proportion | None  # what the simulated users were told (answers given)


def summarise(reps: Sequence[Replicate]) -> Summary:
    lat = [r.latency_days / r.latency_changes for r in reps if r.latency_changes]
    ch = sum(r.latency_changes for r in reps)
    metrics = tuple(
        (m, pooled(reps, m), interval([v for v in rep_values(reps, m) if v is not None]))
        for m in METRICS
    )
    probes = sum(r.total.n for r in reps)
    un = sum(dict(r.unresolved).get(k, 0) for r in reps for k, _ in r.unresolved)
    claims = sum(r.claims for r in reps)
    users: dict[str, int] = defaultdict(int)
    for r in reps:
        for k, v in r.user:
            users[k] += v
    answered = users["correct"] + users["wrong"]
    return Summary(
        world=reps[0].world,
        system=reps[0].system,
        replicates=len(reps),
        metrics=metrics,
        latency_days=interval(lat),
        latency_censored=Proportion.of(sum(r.latency_censored for r in reps), ch, CONFIDENCE)
        if ch
        else None,
        probes=probes,
        items_per_probe_available=quantize(sum(r.total.available for r in reps) / probes)
        if probes
        else None,
        scanned_per_probe=quantize(sum(r.scanned for r in reps) / probes) if probes else None,
        events=sum(r.events for r in reps),
        importance_evals=sum(r.importance_evals for r in reps),
        misattributed=(
            Proportion.of(
                sum(r.misattributed for r in reps), sum(r.claims for r in reps), CONFIDENCE
            )
            if claims
            else None
        ),
        unresolved=Proportion.of(un, un + sum(r.items for r in reps), CONFIDENCE)
        if un + sum(r.items for r in reps)
        else None,
        user_correct=Proportion.of(users["correct"], answered, CONFIDENCE) if answered else None,
    )


class CurvePoint(Record):
    horizon: float
    pooled: Proportion | None
    across: Interval


class Curve(Record):
    world: str
    system: str
    metric: str
    points: tuple[CurvePoint, ...]


def curve(reps: Sequence[Replicate], metric: str) -> Curve:
    pts = tuple(
        CurvePoint(
            horizon=h,
            pooled=pooled(reps, metric, h),
            across=interval([v for v in rep_values(reps, metric, h) if v is not None]),
        )
        for h in CHECKPOINTS
    )
    return Curve(world=reps[0].world, system=reps[0].system, metric=metric, points=pts)


class Comparison(Record):
    """One paired comparison across replicates within a Holm family."""

    world: str
    metric: str
    treatment: str
    baseline: str
    diff: PairedMean  # treatment - baseline
    holm_p: float
    verdict: Literal["worse", "better", "no_detected_difference", "underpowered"]


def _verdict(
    pm: PairedMean, adj: float, harm_positive: bool
) -> Literal["worse", "better", "no_detected_difference", "underpowered"]:
    if pm.mean is None or pm.underpowered:
        return "underpowered"
    if adj < ALPHA and pm.mean != 0:
        return "worse" if (pm.mean > 0) == harm_positive else "better"
    return "no_detected_difference"


def family(
    world: str, metric: str, pairs: Sequence[tuple[str, str]], data: dict[str, list[Replicate]]
) -> tuple[Comparison, ...]:
    harm_positive = metric in ("stale_rate", "wrong_rate", "none_rate")
    todo = [(t, b) for t, b in pairs if t in data and b in data]
    pms = [paired(rep_values(data[t], metric), rep_values(data[b], metric)) for t, b in todo]
    adj = holm([p.sign_p for p in pms])
    return tuple(
        Comparison(
            world=world,
            metric=metric,
            treatment=t,
            baseline=b,
            diff=pm,
            holm_p=significant(a),
            verdict=_verdict(pm, a, harm_positive),
        )
        for (t, b), pm, a in zip(todo, pms, adj, strict=True)
    )


class Onset(Record):
    """When (over time) a treatment's stale rate first exceeds the baseline's."""

    world: str
    treatment: str
    baseline: str
    points: tuple[tuple[float, PairedMean, float], ...]  # horizon, paired diff, Holm p
    onset: float | None  # earliest horizon with a significant increase, else None


def onset(world: str, treatment: str, baseline: str, data: dict[str, list[Replicate]]) -> Onset:
    diffs = [
        paired(
            rep_values(data[treatment], "stale_rate", h),
            rep_values(data[baseline], "stale_rate", h),
        )
        for h in CHECKPOINTS
    ]
    adj = holm([d.sign_p for d in diffs])
    pts = tuple((h, d, significant(a)) for h, d, a in zip(CHECKPOINTS, diffs, adj, strict=True))
    hit = [h for h, d, a in pts if _verdict(d, a, True) == "worse"]
    return Onset(
        world=world,
        treatment=treatment,
        baseline=baseline,
        points=pts,
        onset=min(hit) if hit else None,
    )


class DoseOnset(Record):
    """The smallest dose at which the full-horizon stale rate is significantly above baseline."""

    world: str
    family: str  # "rank_weight" or "gate_threshold"
    baseline: str
    doses: tuple[tuple[float, str, PairedMean, float, float | None], ...]
    # dose, system, paired stale-rate difference, Holm p, pooled item retention of the system
    onset: float | None


def dose_onset(
    world: str,
    name: str,
    doses: Sequence[tuple[float, str]],
    baseline: str,
    data: dict[str, list[Replicate]],
) -> DoseOnset:
    use = [(d, s) for d, s in doses if s in data and s != baseline]
    diffs = [
        paired(rep_values(data[s], "stale_rate"), rep_values(data[baseline], "stale_rate"))
        for _, s in use
    ]
    adj = holm([d.sign_p for d in diffs])
    rows = tuple(
        (
            d,
            s,
            pm,
            significant(a),
            (p.estimate if (p := pooled(data[s], "item_retention")) else None),
        )
        for (d, s), pm, a in zip(use, diffs, adj, strict=True)
    )
    hit = [d for d, _, pm, a, _r in rows if _verdict(pm, a, True) == "worse"]
    return DoseOnset(
        world=world, family=name, baseline=baseline, doses=rows, onset=min(hit) if hit else None
    )


class Ablation(Record):
    """The adaptive system with one importance component removed, against the full model."""

    world: str
    component: str
    stale: Comparison
    correct: Comparison


# --- importance calibration ---------------------------------------------------------------


class ImportanceCalibration(Record):
    """Does importance predict *use* (descriptive), and does it predict *truth* (epistemic)?

    Fitted on calibration replicates (seed family 2), evaluated on test replicates (family 1)
    (I74). ``auroc_use``: P(score of an item used again within 30 days > score of one not);
    ``auroc_truth``: the same against "the item states the current truth"."""

    world: str
    system: str
    n_fit: int
    n_test: int
    base_rate_use: float | None
    auroc_use: float | None
    auroc_truth: float | None
    raw_ece: float | None
    recalibrated_ece: float | None
    raw_brier: float | None
    recalibrated_brier: float | None


def calibration_record(
    world: str,
    system: str,
    fit: list[tuple[float, bool, bool]],
    test: list[tuple[float, bool, bool]],
) -> ImportanceCalibration:
    use_fit = [(s, u) for s, u, _ in fit]
    use = [(s, u) for s, u, _ in test]
    truth = [(s, t) for s, _, t in test]
    model: Isotonic | None = fit_isotonic(use_fit) if use_fit else None
    raw = reliability(use)
    rec = reliability([(model(s), u) for s, u in use]) if model and use else None
    return ImportanceCalibration(
        world=world,
        system=system,
        n_fit=len(fit),
        n_test=len(test),
        base_rate_use=quantize(sum(u for _, u in use) / len(use)) if use else None,
        auroc_use=auroc(use),
        auroc_truth=auroc(truth),
        raw_ece=raw.ece,
        recalibrated_ece=rec.ece if rec else None,
        raw_brier=raw.brier,
        recalibrated_brier=rec.brier if rec else None,
    )


# --- trace audit --------------------------------------------------------------------------


class TraceAudit(Record):
    """Every availability decision and importance record at the checkpoints of one run,
    re-derived from the ledger. ``violations`` must be empty."""

    world: str
    system: str
    decisions: int
    importances: int
    events: int
    cutoff_violations: int
    reproduction_violations: int
    unresolved_importance_links: int
    archived: int
    reasons: tuple[tuple[str, int], ...]


def trace_audit(world: World, system: System, claims: dict[str, tuple[Claim, ...]]) -> TraceAudit:
    res = simulate(world, system, claims, trace=True)
    assert res.ledger is not None
    have = {r.digest for r in res.importances}
    reasons: dict[str, int] = defaultdict(int)
    bad_cut = sum(d.cutoff != d.at for d in res.decisions)
    bad_link = 0
    for d in res.decisions:
        reasons[d.reason] += 1
        if d.importance is not None and d.importance not in have:
            bad_link += 1
    bad_repro = 0
    for r in res.importances:
        it = res.catalog[r.item]
        bad_repro += (
            len(verify_importance(r, res.ledger, it.recorded, it.prior, system.importance)) > 0
        )
    return TraceAudit(
        world=world.name,
        system=system.name,
        decisions=len(res.decisions),
        importances=len(res.importances),
        events=res.events,
        cutoff_violations=bad_cut,
        reproduction_violations=bad_repro,
        unresolved_importance_links=bad_link,
        archived=sum(d.state == "archived" for d in res.decisions),
        reasons=tuple(sorted(reasons.items())),
    )


# --- the study ----------------------------------------------------------------------------


class StudySpec(Record):
    """Everything that determines the study. Its digest is its identity."""

    name: str
    replicates: int = Field(ge=2)
    cal_replicates: int = Field(ge=1)
    worlds: tuple[Digest, ...]  # test then calibration world specs
    systems: tuple[Digest, ...]
    window_days: float
    checkpoints: tuple[float, ...]
    warmup_days: float
    confidence: float
    alpha: float
    identity_policies: tuple[Digest, ...]
    importance: Digest
    world_names: tuple[str, ...]


class CaseOutcome(Record):
    family: Literal["extraction", "identity", "merge"]
    name: str
    policy: str
    passed: bool
    got: str


class AdaptiveStudy(Record):
    spec: Digest
    runs: tuple[tuple[str, str, int, Digest], ...]  # world, system, replicate, Replicate digest
    summaries: tuple[Summary, ...]
    curves: tuple[Curve, ...]
    comparisons: tuple[Comparison, ...]
    onsets: tuple[Onset, ...]
    dose_onsets: tuple[DoseOnset, ...]
    ablations: tuple[Ablation, ...]
    calibrations: tuple[ImportanceCalibration, ...]
    audits: tuple[TraceAudit, ...]
    extraction: tuple[ExtractionStats, ...]
    identity: tuple[IdentityStats, ...]
    cases: tuple[CaseOutcome, ...]


def study_spec(
    store: ArtifactStore,
    replicates: int = 12,
    cal_replicates: int = 6,
    world_names: Sequence[str] = WORLD_NAMES,
) -> StudySpec:
    specs = [
        s.digest
        for base, n in ((1, replicates), (2, cal_replicates))
        for r in range(n)
        for s in world_specs(base, r)
        if s.name in world_names
    ]
    return StudySpec(
        name="super-phase-5-adaptive-memory",
        replicates=replicates,
        cal_replicates=cal_replicates,
        worlds=tuple(specs),
        systems=tuple(store.put_record(s) for _, s in sorted(systems().items())),
        window_days=WINDOW,
        checkpoints=CHECKPOINTS,
        warmup_days=WARMUP_DAYS,
        confidence=CONFIDENCE,
        alpha=ALPHA,
        identity_policies=tuple(store.put_record(p) for p in (CONSERVATIVE, SIMILARITY)),
        importance=store.put_record(DEFAULT),
        world_names=tuple(world_names),
    )


def _policy(system: System) -> IdentityPolicy:
    return SIMILARITY if system.identity == "similarity" else CONSERVATIVE


class Worlds:
    """Generated worlds and their extractions, built once per (base seed, replicate, name)."""

    def __init__(self) -> None:
        self._worlds: dict[tuple[int, int, str], World] = {}
        self._claims: dict[tuple[int, int, str, str], dict[str, tuple[Claim, ...]]] = {}

    def world(self, base: int, rep: int, name: str) -> World:
        key = (base, rep, name)
        if key not in self._worlds:
            spec = next(s for s in world_specs(base, rep) if s.name == name)
            self._worlds[key] = build_world(spec)
        return self._worlds[key]

    def claims(
        self, base: int, rep: int, name: str, policy: IdentityPolicy
    ) -> dict[str, tuple[Claim, ...]]:
        key = (base, rep, name, policy.name)
        if key not in self._claims:
            self._claims[key] = precompute_claims(self.world(base, rep, name), policy)
        return self._claims[key]


def run_replicate(
    cache: Worlds, base: int, rep: int, name: str, system: System, shadow: bool = False
) -> tuple[Replicate, SimResult]:
    w = cache.world(base, rep, name)
    res = simulate(
        w,
        system,
        cache.claims(base, rep, name, _policy(system)),
        shadow=DEFAULT if shadow else None,
    )
    return replicate_record(w, system, rep, res), res


def run_study(
    spec: StudySpec, store: ArtifactStore, log: Callable[[str], None] | None = None
) -> tuple[AdaptiveStudy, PerformanceRecord]:
    say = log or (lambda _m: None)
    cache = Worlds()
    table = systems()
    names = spec.world_names
    timings: dict[str, float] = defaultdict(float)
    data: dict[str, dict[str, list[Replicate]]] = {w: defaultdict(list) for w in names}
    fit_pairs: dict[tuple[str, str], list[tuple[float, bool, bool]]] = defaultdict(list)
    test_pairs: dict[tuple[str, str], list[tuple[float, bool, bool]]] = defaultdict(list)
    runs: list[tuple[str, str, int, str]] = []
    extraction: dict[tuple[str, str], list[ExtractionStats]] = defaultdict(list)
    identity: dict[tuple[str, str], list[IdentityStats]] = defaultdict(list)
    for rep in range(spec.replicates):
        for wn in names:
            for sname, system in sorted(table.items()):
                t0 = time.perf_counter()
                record, res = run_replicate(cache, 1, rep, wn, system, shadow=sname in CAL_SYSTEMS)
                timings[sname] += time.perf_counter() - t0
                data[wn][sname].append(record)
                runs.append((wn, sname, rep, store.put_record(record)))
                if sname in CAL_SYSTEMS:
                    test_pairs[(wn, sname)].extend(res.calibration)
            w = cache.world(1, rep, wn)
            for pol in (CONSERVATIVE, SIMILARITY):
                extraction[(wn, pol.name)].append(
                    extraction_stats(w, cache.claims(1, rep, wn, pol), pol.name)
                )
            identity[(wn, "all")].extend(identity_stats(w, cache.claims(1, rep, wn, CONSERVATIVE)))
        say(f"replicate {rep + 1}/{spec.replicates}")
    for rep in range(spec.cal_replicates):
        for wn in names:
            for sname in CAL_SYSTEMS:
                _, res = run_replicate(cache, 2, rep, wn, table[sname], shadow=True)
                fit_pairs[(wn, sname)].extend(res.calibration)
    summaries: list[Summary] = []
    curves: list[Curve] = []
    comps: list[Comparison] = []
    onsets: list[Onset] = []
    doses: list[DoseOnset] = []
    abl: list[Ablation] = []
    cals: list[ImportanceCalibration] = []
    audits: list[TraceAudit] = []
    for wn in names:
        d = {k: sorted(v, key=lambda r: r.replicate) for k, v in data[wn].items()}
        for sname in sorted(d):
            summaries.append(summarise(d[sname]))
            for m in CURVE_METRICS:
                curves.append(curve(d[sname], m))
        for m in ("stale_rate", "correct_rate", "none_rate"):
            comps.extend(family(wn, m, COMPARISONS, d))
        for t, base in ONSETS:
            onsets.append(onset(wn, t, base, d))
        doses.append(
            dose_onset(
                wn, "rank_weight", list(zip(DOSE_LAMBDAS, RANK_DOSE[1:], strict=True)), BASELINE, d
            )
        )
        doses.append(
            dose_onset(
                wn,
                "gate_threshold",
                list(zip(DOSE_THETAS, GATE_DOSE, strict=True)),
                "B:adaptive-rank",
                d,
            )
        )
        for x in ABLATIONS:
            pair = [(x, ADAPTIVE)]
            (s,) = family(wn, "stale_rate", pair, d)
            (c,) = family(wn, "correct_rate", pair, d)
            abl.append(Ablation(world=wn, component=x.removeprefix("X:minus-"), stale=s, correct=c))
        for sname in CAL_SYSTEMS:
            cals.append(
                calibration_record(wn, sname, fit_pairs[(wn, sname)], test_pairs[(wn, sname)])
            )
        for sname in ("F:gate-adaptive", "G:gate-stale-aware", "B:adaptive-rank"):
            system = table[sname]
            audits.append(
                trace_audit(cache.world(1, 0, wn), system, cache.claims(1, 0, wn, _policy(system)))
            )
        say(f"analysed {wn}")
    ext = tuple(
        pool_extraction(extraction[(wn, p.name)], wn, p.name)
        for wn in names
        for p in (CONSERVATIVE, SIMILARITY)
    )
    ids = tuple(
        pool_identity([s for s in identity[(wn, "all")] if s.policy == p], wn, p)
        for wn in names
        for p in ("conservative", "exact-only", "similarity")
    )
    cases = tuple(
        [
            CaseOutcome(family="extraction", name=n, policy="rules", passed=ok, got=repr(got))
            for n, ok, got in run_extraction_cases()
        ]
        + [
            CaseOutcome(
                family="identity",
                name=n,
                policy=p,
                passed=ok,
                got=f"{got}{' FALSE-MERGE' if fm else ''}",
            )
            for n, p, ok, got, fm in run_identity_cases()
        ]
        + [
            CaseOutcome(
                family="merge",
                name=n,
                policy="min-2-roots",
                passed=True,
                got=f"{d.outcome}: {d.reason}{' (FALSE MERGE)' if fm else ''}",
            )
            for n, d, fm in merge_cases()
        ]
    )
    study = AdaptiveStudy(
        spec=store.put_record(spec),
        runs=tuple(sorted(runs)),
        summaries=tuple(summaries),
        curves=tuple(curves),
        comparisons=tuple(comps),
        onsets=tuple(onsets),
        dose_onsets=tuple(doses),
        ablations=tuple(abl),
        calibrations=tuple(cals),
        audits=tuple(audits),
        extraction=ext,
        identity=ids,
        cases=cases,
    )
    store.put_record(study)
    perf = PerformanceRecord(
        experiment=study.digest,
        environment=environment(),
        timings=tuple((k, significant(v)) for k, v in sorted(timings.items())),
    )
    store.put_record(perf)
    return study, perf


class Reproduction(Record):
    runs_checked: int
    identical: int
    differing: tuple[str, ...]


def reproduce(
    study: AdaptiveStudy,
    spec: StudySpec,
    fresh: ArtifactStore,
    replicates: Sequence[int] = (0,),
) -> Reproduction:
    """Re-simulate the runs of ``replicates`` from the specs alone into an independent store and
    compare digests with the study's."""
    cache = Worlds()
    table = systems()
    want = {(w, s, r): d for w, s, r, d in study.runs}
    bad: list[str] = []
    n = 0
    for rep in replicates:
        for wn in spec.world_names:
            for sname, system in sorted(table.items()):
                record, _ = run_replicate(cache, 1, rep, wn, system, shadow=sname in CAL_SYSTEMS)
                n += 1
                if fresh.put_record(record) != want[(wn, sname, rep)]:
                    bad.append(f"{wn}/{sname}/{rep}")
    return Reproduction(runs_checked=n, identical=n - len(bad), differing=tuple(bad))


__all__ = [
    "ADAPTIVE",
    "BASELINE",
    "AdaptiveStudy",
    "Replicate",
    "StudySpec",
    "Summary",
    "reproduce",
    "run_study",
    "study_spec",
    "systems",
]
