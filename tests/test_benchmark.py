import pytest

from memoria.adaptive_study import systems
from memoria.adaptive_worlds import build_world
from memoria.artifacts import ArtifactStore
from memoria.autopsy_demo import demonstrate
from memoria.belief_worlds import belief_world
from memoria.benchmark import (
    BASELINE,
    ENVIRONMENTS,
    JITTER,
    MEMORY_CAL,
    MEMORY_SYSTEMS,
    MEMORY_TEST,
    BeliefContext,
    BenchRun,
    analyse,
    belief_run,
    belief_spec,
    bench_spec,
    compare_family,
    make_manifest,
    memory_run,
    rate,
)
from memoria.retention import precompute_claims


def test_manifest_is_deterministic_complete_and_uses_new_seeds(
    tmp_path: pytest.TempPathFactory,
) -> None:
    a = make_manifest(ArtifactStore(str(tmp_path) + "/a"), 3, 2, 2, 1)
    b = make_manifest(ArtifactStore(str(tmp_path) + "/b"), 3, 2, 2, 1)
    assert a.digest == b.digest
    assert len(a.memory_worlds) == len(ENVIRONMENTS) * (3 + 2)
    assert len(a.belief_worlds) == len(ENVIRONMENTS) * (2 + 1)
    assert len(a.memory_systems) == len(MEMORY_SYSTEMS)
    assert {n for n, _, _ in a.jitter} == {n for n, _, _ in JITTER}
    assert len({w[3] for w in a.memory_worlds}) == len(a.memory_worlds)  # every world is distinct
    assert make_manifest(ArtifactStore(str(tmp_path) + "/c"), 4, 2, 2, 1).digest != a.digest


def test_worlds_are_deterministic_jittered_and_independent_of_earlier_studies() -> None:
    spec = bench_spec(MEMORY_TEST, 0, "changing")
    assert spec.digest == bench_spec(MEMORY_TEST, 0, "changing").digest
    assert build_world(spec).digest == build_world(bench_spec(MEMORY_TEST, 0, "changing")).digest
    assert spec.collisions
    base_ce = 45
    assert base_ce * 0.75 <= spec.change_every <= base_ce * 1.25
    assert spec.seed not in {1000 + r for r in range(50)} | {2000 + r for r in range(50)}
    assert bench_spec(MEMORY_CAL, 0, "changing").seed != spec.seed
    varied = {bench_spec(MEMORY_TEST, r, "changing").change_every for r in range(6)}
    assert len(varied) == 6


def test_a_memory_run_is_deterministic_and_measures_what_it_claims() -> None:
    w = build_world(bench_spec(MEMORY_TEST, 0, "changing"))
    cl = precompute_claims(w)
    sysm = systems()["G:gate-stale-aware"]
    one, _ = memory_run(w, sysm, "drifting", 0, cl)
    two, _ = memory_run(w, sysm, "drifting", 0, cl)
    assert one.digest == two.digest
    c = dict(one.counts)
    assert c["correct"] + c["stale"] + c["wrong"] + c["none"] == c["n"]
    assert c["autopsy_n"] > 0
    assert (
        c["autopsy_complete"] == c["autopsy_verified"] == c["autopsy_same_answer"] == c["autopsy_n"]
    )
    assert c["replay_ok"] == c["replay_checked"] > 0
    assert 0 < c["contested_n"] <= c["n"]
    assert rate(one, "correct_rate") == c["correct"] / c["n"]


def test_a_belief_run_reports_correctness_calibration_traces_and_replay() -> None:
    w = belief_world(belief_spec(7, 0, "stable"))
    ctx = BeliefContext(w)
    rec, preds = belief_run(ctx, "B:source-weighted", "stable", 0, None)
    c = dict(rec.counts)
    assert c["n"] == len(preds)
    assert c["trace_n"] > 0
    assert c["trace_complete"] == c["trace_n"]  # I76 holds on the sampled beliefs
    assert c["replay_ok"] == c["replay_checked"]
    assert "ece_raw" in dict(rec.values)
    assert belief_run(ctx, "B:source-weighted", "stable", 0, None)[0].digest == rec.digest


def _run(system: str, rep: int, correct: int, n: int = 100) -> BenchRun:
    return BenchRun(
        track="memory",
        environment="drifting",
        system=system,
        replicate=rep,
        seed=rep,
        world="sha256:" + "0" * 64,
        counts=(("correct", correct), ("n", n)),
    )


def test_paired_comparison_is_by_replicate_with_holm_and_a_verdict() -> None:
    runs = [_run(BASELINE, r, 50) for r in range(12)]
    runs += [_run("B:adaptive-rank", r, 60 + r % 2) for r in range(12)]
    runs += [_run("C:age-30", r, 50 + (r % 3 - 1)) for r in range(12)]
    by: dict[str, dict[tuple[str, int], BenchRun]] = {}
    for r in runs:
        by.setdefault(r.system, {})[(r.environment, r.replicate)] = r
    comps = {
        c.treatment: c for c in compare_family("memory", "drifting", "correct_rate", BASELINE, by)
    }
    assert comps["B:adaptive-rank"].verdict == "better"
    assert comps["B:adaptive-rank"].diff.positive == 12
    assert comps["C:age-30"].verdict == "no_detected_difference"
    assert all(c.holm_p >= c.diff.sign_p - 1e-12 for c in comps.values())
    cells, all_comps = analyse(runs, ())
    assert {c.system for c in cells} == {BASELINE, "B:adaptive-rank", "C:age-30"}
    assert any(c.environment == "ALL" for c in all_comps)


def test_the_autopsy_demonstration_reproduces_into_an_independent_store(
    tmp_path: pytest.TempPathFactory,
) -> None:
    one = demonstrate(ArtifactStore(str(tmp_path) + "/x"))
    two = demonstrate(ArtifactStore(str(tmp_path) + "/y"))
    assert one.digest == two.digest
    assert one.memory_autopsy.complete
    assert one.belief_autopsy.subject == "belief"
    assert one.belief_autopsy.complete == (not one.belief_autopsy.missing)
    assert all("not a causal" in c.interpretation for c in one.counterfactuals)
    assert all(c.changed > 0 for c in one.counterfactuals)
