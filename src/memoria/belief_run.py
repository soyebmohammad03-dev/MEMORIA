"""Runs the whole belief-revision and calibration study: ``python -m memoria.belief_run``."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence

from pydantic import Field

from memoria.artifacts import ArtifactStore
from memoria.belief_cases import (
    CaseResult,
    TaxonomyCase,
    run_cases,
    taxonomy_cases,
)
from memoria.belief_demo import Demonstration, Survey, demonstration, survey
from memoria.belief_lab import SELECTIVE, predictions, run_world
from memoria.belief_ops import (
    OpsCell,
    RetrievalComparison,
    ops_study,
    retrieval_vs_belief,
)
from memoria.belief_study import (
    WORLD_NAMES,
    MatrixResult,
    OrderResult,
    Runner,
    SelectiveResult,
    Stability,
    TrustResult,
    matrix,
    order_study,
    selective_study,
    stability,
    trust_study,
)
from memoria.beliefs import POLICIES
from memoria.core import Digest, Record
from memoria.semantic_eval import PerformanceRecord, environment
from memoria.statistics import significant


class StudySpec(Record):
    """Everything that determines the study. Its digest is its identity."""

    name: str = Field(min_length=1)
    replicates: int = Field(ge=1)
    cal_replicates: int = Field(ge=1)
    ops_replicates: int = Field(ge=1)
    orders: int = Field(ge=2)
    max_delay_days: float
    policies: tuple[Digest, ...]
    selective: Digest
    worlds: tuple[Digest, ...]  # world specs of the test replicates, then the calibration ones
    demonstration_world: str
    worlds_run: tuple[str, ...]


class BeliefStudy(Record):
    spec: Digest
    matrix: MatrixResult
    stability: tuple[Stability, ...]
    orders: tuple[OrderResult, ...]
    trust: tuple[TrustResult, ...]
    selective: tuple[SelectiveResult, ...]
    surveys: tuple[Survey, ...]
    taxonomy: tuple[TaxonomyCase, ...]
    cases: tuple[CaseResult, ...]
    ops: tuple[OpsCell, ...]
    retrieval: tuple[RetrievalComparison, ...]
    demonstration: Digest


def study_spec(
    store: ArtifactStore, replicates: int = 4, cal_replicates: int = 4, ops_replicates: int = 2,
    orders: int = 10, world_names: Sequence[str] = WORLD_NAMES,
) -> StudySpec:  # fmt: skip
    from memoria.belief_worlds import worlds

    specs = [
        s.digest
        for base, n in ((1, replicates), (2, cal_replicates))
        for r in range(n)
        for s in worlds(base, r)
        if s.name in world_names
    ]
    return StudySpec(
        name="phase11-12-beliefs-calibration", replicates=replicates, cal_replicates=cal_replicates,
        ops_replicates=ops_replicates, orders=orders, max_delay_days=20.0,
        policies=tuple(store.put_record(POLICIES[n]) for n in sorted(POLICIES)),
        selective=store.put_record(SELECTIVE), worlds=tuple(specs),
        demonstration_world="delayed-correction", worlds_run=tuple(world_names),
    )  # fmt: skip


def run_study(
    spec: StudySpec, store: ArtifactStore, log: Callable[[str], None] | None = None
) -> tuple[BeliefStudy, PerformanceRecord]:  # fmt: skip
    say = log or (lambda _m: None)
    timings: list[tuple[str, float]] = []

    def stage(name: str, t0: float) -> None:
        timings.append((name, time.perf_counter() - t0))
        say(f"{name}: {timings[-1][1]:.0f}s")

    runner = Runner(spec.replicates, spec.cal_replicates)
    names = spec.worlds_run
    t0 = time.perf_counter()
    mx = matrix(runner, store, world_names=names)
    stage("matrix", t0)
    calibrators = dict(mx.calibrators)
    t0 = time.perf_counter()
    stab: list[Stability] = []
    trust: list[TrustResult] = []
    orders: list[OrderResult] = []
    surveys: list[Survey] = []
    for wn in names:
        w = runner.world("test", wn, 0)
        for pname in sorted(POLICIES):
            run = run_world(w, POLICIES[pname])
            stab.append(stability(run, w, pname))
            if pname == "B:source-weighted":
                trust.append(trust_study(w, run))
        surveys.append(survey(w, store)[0])
    stage("stability, trust, survey", t0)
    t0 = time.perf_counter()
    for wn in names:
        w = runner.world("test", wn, 0)
        for pname in sorted(POLICIES):
            orders.append(order_study(w, POLICIES[pname], spec.orders, spec.max_delay_days))
        say(f"orders {wn}")
    stage("orders", t0)
    t0 = time.perf_counter()
    ops = ops_study(
        [runner.world("test", wn, r) for wn in names for r in range(spec.ops_replicates)],
        calibrators, store,
    )  # fmt: skip
    stage("ops", t0)
    t0 = time.perf_counter()
    retrieval = []
    for wn in names:
        w = runner.world("test", wn, 0)
        preds = predictions(run_world(w, POLICIES["B:source-weighted"]), w)
        retrieval.append(retrieval_vs_belief(w, preds))
    stage("retrieval", t0)
    t0 = time.perf_counter()
    dw = runner.world("test", spec.demonstration_world, 0)
    demo = demonstration(dw, calibrators, store)
    stage("demonstration", t0)
    study = BeliefStudy(
        spec=store.put_record(spec), matrix=mx, stability=tuple(stab), orders=tuple(orders),
        trust=tuple(trust), selective=selective_study([(c.policy, c.summary) for c in mx.pooled]),
        surveys=tuple(surveys), taxonomy=taxonomy_cases(), cases=run_cases(), ops=ops,
        retrieval=tuple(retrieval), demonstration=demo.digest,
    )  # fmt: skip
    store.put_record(study)
    perf = PerformanceRecord(experiment=study.digest, environment=environment(),
                             timings=tuple((n, significant(v)) for n, v in timings))  # fmt: skip
    store.put_record(perf)
    return study, perf


class Reproduction(Record):
    ledgers: int
    identical: int
    demonstration_identical: bool
    differing: tuple[str, ...]


def reproduce(study: BeliefStudy, spec: StudySpec, store: ArtifactStore,
              fresh: ArtifactStore) -> Reproduction:  # fmt: skip
    """Rebuild every stored test ledger and the demonstration into an independent store and
    compare digests. Nothing is read back from ``store`` except what the study names."""
    runner = Runner(spec.replicates, spec.cal_replicates)
    bad: list[str] = []
    for wn, pname, digest in study.matrix.ledgers:
        w = runner.world("test", wn, 0)
        got = fresh.put_record(run_world(w, POLICIES[pname]).ledger)
        if got != digest:
            bad.append(f"{wn}/{pname}")
    dw = runner.world("test", spec.demonstration_world, 0)
    demo = demonstration(dw, dict(study.matrix.calibrators), fresh)
    return Reproduction(
        ledgers=len(study.matrix.ledgers), identical=len(study.matrix.ledgers) - len(bad),
        demonstration_identical=demo.digest == study.demonstration, differing=tuple(bad),
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    from memoria.belief_report import report

    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    quick = "--quick" in args
    spec = (
        study_spec(store, 1, 1, 1, 3, ("stable", "contradiction", "duplicated-evidence"))
        if quick
        else study_spec(store)
    )

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    study, perf = run_study(spec, store, say)
    fresh = ArtifactStore(str(store.root) + "-fresh")
    rep = reproduce(study, spec, store, fresh)
    say(f"reproduction {rep.identical}/{rep.ledgers} ledgers, demo {rep.demonstration_identical}")
    demo = store.get_record(Demonstration, study.demonstration)
    print(f"spec          {study.spec}")
    print(f"study         {study.digest}")
    print(f"performance   {perf.digest}")
    print(f"demonstration {demo.digest}")
    print(f"reproduction  {rep.model_dump_json()}")
    print()
    print(report(study, demo))


if __name__ == "__main__":
    main()
