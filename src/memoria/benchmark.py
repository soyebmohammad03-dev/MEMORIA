"""Multi-seed benchmark over five environments (Super-Phase 6).

Two tracks share five environment classes (stable, drifting, repetitive, noisy, adversarial):

- **memory**: the Super-Phase 5 systems on generated worlds, answers judged against hidden truth
  by audit probes; measures retrieval accuracy, stale-answer rate, retention, contradiction
  handling (accuracy on probes with concurrent disagreement among known evidence), importance
  calibration and provenance completeness (item -> report text -> claim span);
- **belief**: Phase 11-12 revision policies on belief worlds; measures belief correctness,
  coverage, selective accuracy, accuracy on contradiction-heavy keys, raw and recalibrated
  calibration error, and belief-trace completeness (I76).

Both also record autopsy completeness and replay exactness: a sample of each run's decisions is
explained by :mod:`memoria.autopsy` (memory) or :func:`memoria.belief_demo.belief_trace`
(belief) and replayed from the stored records, and the exact agreement is a measurement.

Generation is deterministic: every world is a function of (seed family, replicate, environment)
and a declared jitter of a few parameters (drawn with ``unit_interval``), the complete
:class:`BenchmarkManifest` (worlds by digest, systems, seeds, jitter, metrics, statistics) is
stored, and each run is a record whose digest a reproduction must match. Test and calibration
seeds are separate families (I74); the seeds are new (not those of earlier phases).

Inference is per replicate (independent seeds): paired difference against the baseline (system
A), Student-t interval, exact sign test, Holm within a family (track x environment x metric)
and an underpowered flag; a pooled family pairs (environment, replicate) units. Nothing here is
a universal claim.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from memoria.adaptive import DEFAULT
from memoria.adaptive_study import (
    ALPHA,
    CONFIDENCE,
    ImportanceCalibration,
    Interval,
    _verdict,
    calibration_record,
    interval,
    systems,
)
from memoria.adaptive_worlds import HORIZON, World, WorldSpec, build_world, world_specs
from memoria.artifacts import ArtifactStore
from memoria.autopsy import (
    historical_mismatches,
    memory_autopsy,
    verify_autopsy,
)
from memoria.belief_demo import GRAPH_POLICY, belief_trace
from memoria.belief_lab import predictions, run_world
from memoria.belief_ops import memory_log
from memoria.belief_worlds import BeliefWorld, BeliefWorldSpec, belief_world
from memoria.belief_worlds import worlds as belief_worlds
from memoria.beliefs import POLICIES
from memoria.calibration import (
    Isotonic,
    Prediction,
    answered_pairs,
    apply,
    fit_isotonic,
    reliability,
)
from memoria.claims import Claim
from memoria.consolidation import consolidate
from memoria.core import Digest, Record, quantize, unit_interval
from memoria.graph import build_graph
from memoria.hybrid import Corpus
from memoria.memory_lab import HIERARCHICAL
from memoria.retention import (
    CONCURRENCY_DAYS,
    WARMUP_DAYS,
    Item,
    SimResult,
    System,
    precompute_claims,
    simulate,
)
from memoria.revision import historical_mismatches as belief_mismatches
from memoria.scenarios import day
from memoria.semantic_eval import PerformanceRecord, environment
from memoria.statistics import PairedMean, Proportion, holm, paired_mean, significant

BENCH_VERSION = "1"
# environment, memory world, belief world
ENVIRONMENTS: tuple[tuple[str, str, str], ...] = (
    ("stable", "stable", "stable"),
    ("drifting", "changing", "temporal-change"),
    ("repetitive", "repetitive", "duplicated-evidence"),
    ("noisy", "noisy", "source-disagreement"),
    ("adversarial", "adversarial", "adversarial"),
)
MEMORY_SYSTEMS = (
    "A:static-latest",
    "B:adaptive-rank",
    "C:age-30",
    "D:age-90",
    "E:gate-latest",
    "F:gate-adaptive",
    "G:gate-stale-aware",
    "H:static-importance",
)
BELIEF_POLICIES = (
    "A:evidence-count",
    "B:source-weighted",
    "C:recency-aware",
    "D:temporal-validity",
    "H:conservative",
)
BASELINE = "A:static-latest"
BELIEF_BASELINE = "A:evidence-count"
JITTER = (("change_every", 0.75, 1.25), ("query_rate", 0.8, 1.2), ("stray", 0.8, 1.2))
MEMORY_TEST, MEMORY_CAL, BELIEF_TEST, BELIEF_CAL = 3, 4, 7, 8  # seed families (new)
CAL_SYSTEMS = ("A:static-latest", "B:adaptive-rank")
AUTOPSY_SAMPLE = 6  # autopsies per memory run
REPLAY_EVERY = 10  # replay every 10th recorded decision of a memory run
BELIEF_TRACES = 5  # belief traces per belief run

MEMORY_METRICS: dict[str, tuple[str, str]] = {
    "correct_rate": ("correct", "n"),
    "stale_rate": ("stale", "n"),
    "none_rate": ("none", "n"),
    "evidence_retention": ("evidence_kept", "evidence_known"),
    "item_retention": ("available", "known"),
    "contested_correct_rate": ("contested_correct", "contested_n"),
    "lineage_complete": ("lineage_ok", "answered"),
    "autopsy_complete": ("autopsy_complete", "autopsy_n"),
    "autopsy_verified": ("autopsy_verified", "autopsy_n"),
    "replay_exact": ("replay_ok", "replay_checked"),
}
BELIEF_METRICS: dict[str, tuple[str, str]] = {
    "correct_rate": ("correct", "n"),
    "coverage": ("answered", "n"),
    "selective_accuracy": ("correct", "answered"),
    "contradiction_correct_rate": ("contradiction_correct", "contradiction_n"),
    "trace_complete": ("trace_complete", "trace_n"),
    "replay_exact": ("replay_ok", "replay_checked"),
}
BELIEF_SCALARS = ("ece_raw", "ece_recal", "brier")
COMPARE = {
    "memory": ("correct_rate", "stale_rate", "evidence_retention", "contested_correct_rate"),
    "belief": ("correct_rate", "selective_accuracy", "contradiction_correct_rate", "ece_raw"),
}
HARM_POSITIVE = ("stale_rate", "none_rate", "ece_raw", "ece_recal", "brier")


class BenchmarkManifest(Record):
    """Everything that determines the benchmark. Its digest is its identity."""

    name: str
    version: str
    environments: tuple[tuple[str, str, str], ...]
    memory_seeds: tuple[int, int, int, int]  # test family, calibration family, replicates, cal
    belief_seeds: tuple[int, int, int, int]
    jitter: tuple[tuple[str, float, float], ...]  # parameter, low, high multiplier
    memory_worlds: tuple[tuple[str, str, int, Digest], ...]  # family, environment, replicate, spec
    belief_worlds: tuple[tuple[str, str, int, Digest], ...]
    memory_systems: tuple[Digest, ...]
    belief_policies: tuple[Digest, ...]
    importance: Digest
    metrics: tuple[tuple[str, tuple[str, ...]], ...]
    statistics: tuple[tuple[str, str], ...]
    sampling: tuple[tuple[str, int], ...]


def bench_spec(base: int, replicate: int, world: str) -> WorldSpec:
    """The memory world ``world`` of one replicate, with declared seeded parameter jitter and the
    near-collision entities added (16 keys)."""
    spec = next(s for s in world_specs(base, replicate) if s.name == world)
    upd: dict[str, object] = {"collisions": True}
    for name, lo, hi in JITTER:
        u = unit_interval(spec.seed, "bench-jitter", world, name)
        cur = getattr(spec, name)
        if cur:
            upd[name] = (
                min(1.0, cur * (lo + (hi - lo) * u))
                if name == "stray"
                else cur * (lo + (hi - lo) * u)
            )
    return WorldSpec.model_validate({**spec.model_dump(), **upd})


def belief_spec(seed: int, replicate: int, world: str) -> BeliefWorldSpec:
    return next(s for s in belief_worlds(seed, replicate) if s.name == world)


def make_manifest(
    store: ArtifactStore,
    replicates: int = 30,
    cal: int = 8,
    belief_reps: int = 10,
    belief_cal: int = 5,
) -> BenchmarkManifest:
    table = systems()
    mw = tuple(
        (fam, env, r, bench_spec(base, r, mem).digest)
        for fam, base, n in (("test", MEMORY_TEST, replicates), ("cal", MEMORY_CAL, cal))
        for env, mem, _ in ENVIRONMENTS
        for r in range(n)
    )
    bw = tuple(
        (fam, env, r, belief_spec(seed, r, bel).digest)
        for fam, seed, n in (("test", BELIEF_TEST, belief_reps), ("cal", BELIEF_CAL, belief_cal))
        for env, _, bel in ENVIRONMENTS
        for r in range(n)
    )
    return BenchmarkManifest(
        name="super-phase-6-benchmark",
        version=BENCH_VERSION,
        environments=ENVIRONMENTS,
        memory_seeds=(MEMORY_TEST, MEMORY_CAL, replicates, cal),
        belief_seeds=(BELIEF_TEST, BELIEF_CAL, belief_reps, belief_cal),
        jitter=JITTER,
        memory_worlds=mw,
        belief_worlds=bw,
        memory_systems=tuple(store.put_record(table[n]) for n in MEMORY_SYSTEMS),
        belief_policies=tuple(store.put_record(POLICIES[n]) for n in BELIEF_POLICIES),
        importance=store.put_record(DEFAULT),
        metrics=(("memory", tuple(MEMORY_METRICS)), ("belief", (*BELIEF_METRICS, *BELIEF_SCALARS))),
        statistics=(
            ("unit", "replicate (independent seed)"),
            ("interval", "student-t, 95%"),
            ("test", "exact sign test on paired replicate differences"),
            (
                "multiplicity",
                "Holm within track x environment x metric; pooled family over (env, replicate)",
            ),
            ("baseline", f"{BASELINE} (memory), {BELIEF_BASELINE} (belief)"),
            ("alpha", str(ALPHA)),
        ),
        sampling=(
            ("autopsies_per_memory_run", AUTOPSY_SAMPLE),
            ("replay_every", REPLAY_EVERY),
            ("belief_traces_per_run", BELIEF_TRACES),
        ),
    )


# --- run records --------------------------------------------------------------------------


class BenchRun(Record):
    track: Literal["memory", "belief"]
    environment: str
    system: str
    replicate: int
    seed: int
    world: Digest
    counts: tuple[tuple[str, int], ...]
    values: tuple[tuple[str, float], ...] = ()  # scalar measurements (calibration error, ...)


def _contested(items: Sequence[Item]) -> bool:
    if not items:
        return False
    top = max(i.occurred for i in items)
    return len({i.value for i in items if i.occurred >= top - CONCURRENCY_DAYS}) > 1


def memory_run(
    world: World,
    system: System,
    env: str,
    rep: int,
    claims: dict[str, tuple[Claim, ...]],
    shadow: bool = False,
) -> tuple[BenchRun, SimResult]:
    res = simulate(world, system, claims, shadow=DEFAULT if shadow else None)
    audits = [a for a in res.audits if a.t > WARMUP_DAYS]
    by_key: dict[str, list[Item]] = defaultdict(list)
    for it in res.catalog.values():
        by_key[it.key].append(it)
    c: dict[str, int] = defaultdict(int)
    for k in ("n", "correct", "stale", "wrong", "none", "contested_n", "contested_correct"):
        c[k] += 0  # every count is present, including zeros
    for a in audits:
        c["n"] += 1
        c[a.status] += 1
        c["answered"] += a.status != "none"
        c["lineage_ok"] += a.lineage_ok is True
        c["evidence_known"] += a.evidence_known
        c["evidence_kept"] += a.evidence_known and a.evidence_available
        c["available"] += a.available
        c["known"] += a.known
        if _contested([i for i in by_key[a.key] if i.recorded <= a.t]):
            c["contested_n"] += 1
            c["contested_correct"] += a.status == "correct"
    answered = [a for a in audits if a.item is not None]
    if answered:
        pick = sorted(
            {
                round(j * (len(answered) - 1) / max(1, AUTOPSY_SAMPLE - 1))
                for j in range(AUTOPSY_SAMPLE)
            }
        )
        for idx in pick:
            a = answered[idx]
            au = memory_autopsy(world, system, res, claims, a.key, a.t, "audit")
            c["autopsy_n"] += 1
            c["autopsy_complete"] += au.complete
            c["autopsy_verified"] += not verify_autopsy(au, world, system, res, claims)
            c["autopsy_same_answer"] += au.links[0].detail[1] == ("item", a.item)
    checked, bad = historical_mismatches(res, system, REPLAY_EVERY)
    c["replay_checked"] = checked
    c["replay_ok"] = checked - len(bad)
    counts = tuple(sorted((k, int(v)) for k, v in c.items()))
    return BenchRun(
        track="memory",
        environment=env,
        system=system.name,
        replicate=rep,
        seed=world.seed,
        world=world.digest,
        counts=counts,
    ), res


class BeliefContext:
    """A belief world with its corpus, consolidation and graph (built once, shared by policies)."""

    def __init__(self, world: BeliefWorld) -> None:
        self.world = world
        with memory_log(world) as log:
            end = day(world.spec.horizon_days + world.spec.delay_days * 2 + 30)
            self.corpus = Corpus.from_log(log, end)
            self.hier = consolidate(self.corpus, HIERARCHICAL, end)
            self.snap = build_graph(
                self.corpus.with_derived(self.hier.memories, ()), GRAPH_POLICY, [self.hier]
            )


def belief_run(
    ctx: BeliefContext, policy: str, env: str, rep: int, calibrator: Isotonic | None
) -> tuple[BenchRun, list[Prediction]]:
    world = ctx.world
    run = run_world(world, POLICIES[policy])
    preds = predictions(run, world)
    n = len(preds)
    ans = [p for p in preds if p.answered and p.correct is not None]
    contra = [p for p in preds if "contradiction-heavy" in p.tags]
    values: dict[str, float] = {}
    rel = reliability(answered_pairs(preds))
    if rel.ece is not None and rel.brier is not None:
        values["ece_raw"], values["brier"] = rel.ece, rel.brier
    if calibrator is not None:
        rc = reliability(answered_pairs(apply(calibrator, preds), calibrated=True))
        if rc.ece is not None:
            values["ece_recal"] = rc.ece
    c = {
        "n": n,
        "answered": len(ans),
        "correct": sum(bool(p.correct) for p in ans),
        "contradiction_n": len(contra),
        "contradiction_correct": sum(bool(p.answered and p.correct) for p in contra),
    }
    by_id = {p.id: p for p in world.probes}
    if ans:
        pick = sorted(
            {round(j * (len(ans) - 1) / max(1, BELIEF_TRACES - 1)) for j in range(BELIEF_TRACES)}
        )
        for idx in pick:
            pr = by_id[ans[idx].probe]
            tr = belief_trace(
                run,
                ctx.corpus,
                ctx.hier,
                ctx.snap,
                pr.key,
                day(pr.valid_at_day),
                day(pr.known_at_day),
            )
            c["trace_n"] = c.get("trace_n", 0) + 1
            c["trace_complete"] = c.get("trace_complete", 0) + tr.complete
    times = [day(world.spec.horizon_days / 2), day(world.spec.horizon_days)]
    bad = belief_mismatches(run, times)
    c["replay_checked"] = len(times) * len(run.history)
    c["replay_ok"] = c["replay_checked"] - bad
    rec = BenchRun(
        track="belief",
        environment=env,
        system=policy,
        replicate=rep,
        seed=world.spec.seed,
        world=world.dataset.digest,
        counts=tuple(sorted(c.items())),
        values=tuple(sorted((k, quantize(v)) for k, v in values.items())),
    )
    return rec, preds


# --- analysis -----------------------------------------------------------------------------


def rate(run: BenchRun, metric: str) -> float | None:
    table = MEMORY_METRICS if run.track == "memory" else BELIEF_METRICS
    if metric in table:
        num, den = table[metric]
        c = dict(run.counts)
        return c.get(num, 0) / c[den] if c.get(den) else None
    return dict(run.values).get(metric)


class BenchCell(Record):
    track: str
    environment: str
    system: str
    replicates: int
    metrics: tuple[tuple[str, Proportion | None, Interval], ...]


class BenchComparison(Record):
    track: str
    environment: str  # or "ALL" (pooled over environment-replicate units)
    metric: str
    treatment: str
    baseline: str
    diff: PairedMean
    holm_p: float
    verdict: Literal["worse", "better", "no_detected_difference", "underpowered"]


def cell(track: str, env: str, system: str, runs: Sequence[BenchRun]) -> BenchCell:
    table = MEMORY_METRICS if track == "memory" else BELIEF_METRICS
    out: list[tuple[str, Proportion | None, Interval]] = []
    for m in (*table, *(BELIEF_SCALARS if track == "belief" else ())):
        vals = [v for v in (rate(r, m) for r in runs) if v is not None]
        pooled: Proportion | None = None
        if m in table:
            num, den = table[m]
            k = sum(dict(r.counts).get(num, 0) for r in runs)
            d = sum(dict(r.counts).get(den, 0) for r in runs)
            pooled = Proportion.of(k, d, CONFIDENCE) if d else None
        out.append((m, pooled, interval(vals)))
    return BenchCell(
        track=track, environment=env, system=system, replicates=len(runs), metrics=tuple(out)
    )


def compare_family(
    track: str,
    env: str,
    metric: str,
    base: str,
    data: dict[str, dict[tuple[str, int], BenchRun]],
) -> tuple[BenchComparison, ...]:
    """Systems against ``base`` on ``metric``, paired by (environment, replicate) unit."""
    harm = metric in HARM_POSITIVE
    rows = []
    for sysname in sorted(s for s in data if s != base):
        diffs = []
        for unit, run in sorted(data[sysname].items()):
            b = data[base].get(unit)
            x, y = rate(run, metric), rate(b, metric) if b else None
            if x is not None and y is not None:
                diffs.append(x - y)
        rows.append((sysname, paired_mean(diffs, CONFIDENCE, ALPHA)))
    adj = holm([pm.sign_p for _, pm in rows])
    return tuple(
        BenchComparison(
            track=track,
            environment=env,
            metric=metric,
            treatment=s,
            baseline=base,
            diff=pm,
            holm_p=significant(a),
            verdict=_verdict(pm, a, harm),
        )
        for (s, pm), a in zip(rows, adj, strict=True)
    )


class BenchResult(Record):
    manifest: Digest
    runs: tuple[
        tuple[str, str, str, int, Digest], ...
    ]  # track, environment, system, replicate, run
    cells: tuple[BenchCell, ...]
    comparisons: tuple[BenchComparison, ...]
    calibrations: tuple[ImportanceCalibration, ...]


def analyse(
    runs: Sequence[BenchRun], calibrations: Sequence[ImportanceCalibration]
) -> tuple[tuple[BenchCell, ...], tuple[BenchComparison, ...]]:
    cells: list[BenchCell] = []
    comps: list[BenchComparison] = []
    for track, base in (("memory", BASELINE), ("belief", BELIEF_BASELINE)):
        tr = [r for r in runs if r.track == track]
        envs = [e for e, _, _ in ENVIRONMENTS]
        for env in (*envs, "ALL"):
            sel = [r for r in tr if env == "ALL" or r.environment == env]
            by_sys: dict[str, dict[tuple[str, int], BenchRun]] = defaultdict(dict)
            for r in sel:
                by_sys[r.system][(r.environment, r.replicate)] = r
            if env != "ALL":
                cells += [
                    cell(track, env, s, [v for _, v in sorted(d.items())])
                    for s, d in sorted(by_sys.items())
                ]
            for m in COMPARE[track]:
                comps.extend(compare_family(track, env, m, base, by_sys))
    return tuple(cells), tuple(comps)


def run_benchmark(
    manifest: BenchmarkManifest,
    store: ArtifactStore,
    log: Callable[[str], None] | None = None,
) -> tuple[BenchResult, PerformanceRecord]:
    say = log or (lambda _m: None)
    table = systems()
    reps, cal_reps = manifest.memory_seeds[2], manifest.memory_seeds[3]
    b_reps, b_cal = manifest.belief_seeds[2], manifest.belief_seeds[3]
    timings: dict[str, float] = defaultdict(float)
    runs: list[BenchRun] = []
    fit: dict[tuple[str, str], list[tuple[float, bool, bool]]] = defaultdict(list)
    test: dict[tuple[str, str], list[tuple[float, bool, bool]]] = defaultdict(list)
    claims_cache: dict[tuple[int, int, str], tuple[World, dict[str, tuple[Claim, ...]]]] = {}

    def memory_world(base: int, rep: int, mem: str) -> tuple[World, dict[str, tuple[Claim, ...]]]:
        k = (base, rep, mem)
        if k not in claims_cache:
            w = build_world(bench_spec(base, rep, mem))
            claims_cache[k] = (w, precompute_claims(w))
        return claims_cache[k]

    for rep in range(reps):
        for env, mem, _ in ENVIRONMENTS:
            w, cl = memory_world(MEMORY_TEST, rep, mem)
            for name in MEMORY_SYSTEMS:
                t0 = time.perf_counter()
                rec, res = memory_run(w, table[name], env, rep, cl, shadow=name in CAL_SYSTEMS)
                timings[f"memory/{name}"] += time.perf_counter() - t0
                store.put_record(rec)
                runs.append(rec)
                if name in CAL_SYSTEMS:
                    test[(env, name)].extend(res.calibration)
            claims_cache.pop((MEMORY_TEST, rep, mem))
        say(f"memory replicate {rep + 1}/{reps}")
    for rep in range(cal_reps):
        for env, mem, _ in ENVIRONMENTS:
            w, cl = memory_world(MEMORY_CAL, rep, mem)
            for name in CAL_SYSTEMS:
                fit[(env, name)].extend(simulate(w, table[name], cl, shadow=DEFAULT).calibration)
            claims_cache.pop((MEMORY_CAL, rep, mem))
    say("memory calibration replicates done")
    calibrators: dict[tuple[str, str], Isotonic | None] = {}
    for env, _, bel in ENVIRONMENTS:
        for p in BELIEF_POLICIES:
            pairs = []
            for r in range(b_cal):
                bw = belief_world(belief_spec(BELIEF_CAL, r, bel))
                pairs += answered_pairs(predictions(run_world(bw, POLICIES[p]), bw))
            calibrators[(env, p)] = fit_isotonic(pairs) if pairs else None
    say("belief calibrators fitted")
    for rep in range(b_reps):
        for env, _, bel in ENVIRONMENTS:
            ctx = BeliefContext(belief_world(belief_spec(BELIEF_TEST, rep, bel)))
            for p in BELIEF_POLICIES:
                t0 = time.perf_counter()
                rec, _ = belief_run(ctx, p, env, rep, calibrators[(env, p)])
                timings[f"belief/{p}"] += time.perf_counter() - t0
                store.put_record(rec)
                runs.append(rec)
        say(f"belief replicate {rep + 1}/{b_reps}")
    cals = tuple(
        calibration_record(env, s, fit[(env, s)], test[(env, s)])
        for env, _, _ in ENVIRONMENTS
        for s in CAL_SYSTEMS
    )
    cells, comps = analyse(runs, cals)
    result = BenchResult(
        manifest=store.put_record(manifest),
        runs=tuple(sorted((r.track, r.environment, r.system, r.replicate, r.digest) for r in runs)),
        cells=cells,
        comparisons=comps,
        calibrations=cals,
    )
    store.put_record(result)
    perf = PerformanceRecord(
        experiment=result.digest,
        environment=environment(),
        timings=tuple((k, significant(v)) for k, v in sorted(timings.items())),
    )
    store.put_record(perf)
    return result, perf


class BenchReproduction(Record):
    runs_checked: int
    identical: int
    differing: tuple[str, ...]
    result_recomputed_identical: bool


def reproduce(
    result: BenchResult,
    manifest: BenchmarkManifest,
    store: ArtifactStore,
    fresh: ArtifactStore,
    memory_reps: Sequence[int],
    belief_reps: Sequence[int],
) -> BenchReproduction:
    """Re-run the given replicates from the manifest alone into an independent store and compare
    every run digest with the study's; then recompute the analysis from the stored runs."""
    want = {(t, e, s, r): d for t, e, s, r, d in result.runs}
    table = systems()
    bad: list[str] = []
    n = 0
    for rep in memory_reps:
        for env, mem, _ in ENVIRONMENTS:
            w = build_world(bench_spec(MEMORY_TEST, rep, mem))
            cl = precompute_claims(w)
            for name in MEMORY_SYSTEMS:
                rec, _ = memory_run(w, table[name], env, rep, cl, shadow=name in CAL_SYSTEMS)
                n += 1
                if fresh.put_record(rec) != want[("memory", env, name, rep)]:
                    bad.append(f"memory/{env}/{name}/{rep}")
    if belief_reps:
        cal: dict[tuple[str, str], Isotonic | None] = {}
        for env, _, bel in ENVIRONMENTS:
            for p in BELIEF_POLICIES:
                pairs = []
                for r in range(manifest.belief_seeds[3]):
                    bw = belief_world(belief_spec(BELIEF_CAL, r, bel))
                    pairs += answered_pairs(predictions(run_world(bw, POLICIES[p]), bw))
                cal[(env, p)] = fit_isotonic(pairs) if pairs else None
        for rep in belief_reps:
            for env, _, bel in ENVIRONMENTS:
                ctx = BeliefContext(belief_world(belief_spec(BELIEF_TEST, rep, bel)))
                for p in BELIEF_POLICIES:
                    rec, _ = belief_run(ctx, p, env, rep, cal[(env, p)])
                    n += 1
                    if fresh.put_record(rec) != want[("belief", env, p, rep)]:
                        bad.append(f"belief/{env}/{p}/{rep}")
    stored = [store.get_record(BenchRun, d) for *_, d in result.runs]
    cells, comps = analyse(stored, result.calibrations)
    same = cells == result.cells and comps == result.comparisons
    return BenchReproduction(
        runs_checked=n,
        identical=n - len(bad),
        differing=tuple(bad),
        result_recomputed_identical=same,
    )


__all__ = ["HORIZON", "BenchResult", "make_manifest", "reproduce", "run_benchmark"]
