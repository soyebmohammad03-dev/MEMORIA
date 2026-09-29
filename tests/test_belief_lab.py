from pathlib import Path

import pytest

from memoria.artifacts import ArtifactStore
from memoria.belief_cases import CaseResult, adversarial_cases, run_case, run_cases
from memoria.belief_demo import GRAPH_POLICY, Demonstration, belief_trace, survey
from memoria.belief_lab import (
    identity_map,
    predictions,
    run_world,
    summarize,
    with_signals,
)
from memoria.belief_ops import (
    CONDITIONS,
    build_condition,
    memory_log,
    ops_study,
    retrieval_vs_belief,
)
from memoria.belief_run import reproduce, run_study, study_spec
from memoria.belief_study import (
    Runner,
    matrix,
    order_study,
    reorder,
    stability,
    trust_study,
)
from memoria.belief_worlds import BeliefWorld, BeliefWorldSpec, belief_world, worlds
from memoria.beliefs import POLICIES
from memoria.calibration import answered_pairs, fit_isotonic
from memoria.consolidation import consolidate
from memoria.core import EpistemicStatus
from memoria.graph import build_graph
from memoria.hybrid import Corpus
from memoria.memory_lab import HIERARCHICAL
from memoria.revision import violations
from memoria.scenarios import day

TINY = BeliefWorldSpec(
    name="tiny", seed=5, entities=2, attributes=("home", "dose"), horizon_days=200,
    change_every_days=60, rival_rate=0.4, copy_rate=0.5, copies=2, correction_rate=0.3,
    delay_days=4.0, probes_per_key=3,
)  # fmt: skip


@pytest.fixture(scope="module")
def world() -> BeliefWorld:
    return belief_world(TINY)


# --- worlds --------------------------------------------------------------------------------------


def test_worlds_are_deterministic_and_keep_exact_ground_truth(world: BeliefWorld) -> None:
    again = belief_world(TINY)
    assert again.dataset.digest == world.dataset.digest
    assert [e.id for e in again.evidence] == [e.id for e in world.evidence]
    assert all(e.recorded_at >= e.occurred_at for e in world.evidence)
    ids = {e.id for e in world.evidence}
    assert world.wrong_reports <= ids
    assert {p.key for p in world.probes} == set(world.truth)
    for p in world.probes:
        assert world.truth_at(p.key, p.valid_at_day) is not None
        assert p.known_at_day >= p.valid_at_day
    # the declared model knows the copies the world made, and nothing about reliabilities
    assert dict(world.sources.copies) == world.origin_of
    assert world.origin_of
    assert all(c.name in ("chat", "clinic", "email", "forum") for c in world.sources.classes)
    other = belief_world(TINY.model_copy(update={"seed": 6}))
    assert other.dataset.digest != world.dataset.digest


def test_the_ten_worlds_exist_and_differ_only_by_seed_across_replicates() -> None:
    a, b = worlds(1, 0), worlds(1, 1)
    assert [s.name for s in a] == [s.name for s in b]
    assert len({s.name for s in a}) == 10
    assert a[0].seed != b[0].seed
    assert {s.name for s in worlds(2)} == {s.name for s in a}
    ambiguous = next(s for s in a if s.name == "ambiguous-entities")
    keys = {k for k in belief_world(ambiguous).truth}
    assert {"ana.home", "anna.home"} <= keys


def test_erroneous_reports_and_copies_are_what_the_ledger_finds_hard(world: BeliefWorld) -> None:
    run = run_world(world, POLICIES["A:evidence-count"])
    finals_a = [kb for k in run.history if (kb := run.final(k)) is not None]
    copied = [b for kb in finals_a for b in kb.beliefs if b.naive_count > b.independent_count]
    assert copied  # some belief rests on copies counted as separate reports
    careful = run_world(world, POLICIES["E:corroboration"])
    finals = [kb for k in careful.history if (kb := careful.final(k)) is not None]
    assert all(b.weight <= b.independent_count for kb in finals for b in kb.beliefs)


# --- predictions and summaries -------------------------------------------------------------------


def test_predictions_are_judged_against_the_hidden_truth(world: BeliefWorld) -> None:
    run = run_world(world, POLICIES["B:source-weighted"])
    preds = predictions(run, world)
    assert [p.probe for p in preds] == [p.id for p in world.probes]
    truth = {p.id: p.truth for p in world.probes}
    for p in preds:
        if p.answered:
            assert p.correct == (p.value == truth[p.probe])
        else:
            assert p.correct is None
            assert p.value is None
    s = summarize(preds)
    assert s.n == len(preds)
    assert s.answered == sum(p.answered for p in preds)
    assert s.correct.numerator == sum(bool(p.correct) for p in preds)
    assert sum(n for _, n in s.modes) == s.n
    assert s.reliability.n == s.answered


def test_a_calibrator_is_recorded_beside_the_score_and_never_replaces_it(
    world: BeliefWorld,
) -> None:
    run = run_world(world, POLICIES["A:evidence-count"])
    raw = predictions(run, world)
    m = fit_isotonic(answered_pairs(raw))
    fitted = predictions(run, world, calibrator=m)
    assert [p.score for p in fitted] == [p.score for p in raw]
    assert [p.mode for p in fitted] == [p.mode for p in raw]
    assert all(p.calibrated is None for p in raw)
    assert all(p.calibrated == m(p.score) for p in fitted if p.score is not None)


def test_all_ten_worlds_pass_the_audit_under_every_policy() -> None:
    for spec in worlds(9, 0):
        w = belief_world(spec.model_copy(update={"horizon_days": 120}))
        for name in ("A:evidence-count", "B:source-weighted", "H:conservative"):
            r = run_world(w, POLICIES[name])
            assert violations(r, arithmetic=False) == [], (spec.name, name)


# --- adversarial cases ---------------------------------------------------------------------------


def case(name: str, policy: str, **kw: bool) -> CaseResult:
    c = next(c for c in adversarial_cases() if c.name == name)
    return run_case(c, POLICIES[policy], **kw)


def test_copied_evidence_fools_counting_but_not_independence() -> None:
    naive, careful = (
        case("copied-evidence", "A:evidence-count"),
        case("copied-evidence", "E:corroboration"),
    )
    assert (naive.value, naive.correct, naive.overconfident) == ("rome", False, True)
    assert (naive.naive, naive.independent) == (6, 1)
    assert careful.mode == "competing"
    assert not careful.overconfident


def test_many_weak_sources_are_independent_and_the_negative_result_is_recorded() -> None:
    for policy in ("A:evidence-count", "E:corroboration", "H:conservative"):
        r = case("weak-flood", policy)
        assert r.value == "rome", policy  # six independent weak reports outweigh one strong one
        assert r.overconfident
    weighted = case("weak-flood", "B:source-weighted").score
    counted = case("weak-flood", "A:evidence-count").score
    assert weighted is not None
    assert counted is not None
    assert weighted < counted


def test_a_summary_cannot_outlive_its_evidence_without_saying_so() -> None:
    for name in POLICIES:
        naive = case("weakened-derived", name)
        aware = case("weakened-derived", name, lineage=True)
        assert not aware.overconfident, name
        assert aware.mode in ("require_evidence", "abstain", "competing"), name
        if name in ("A:evidence-count", "B:source-weighted", "D:temporal-validity"):
            assert naive.overconfident  # confident in a belief nothing supports any more


def test_lineage_stops_a_summary_of_copies_counting_as_five() -> None:
    naive = case("misleading-summary", "A:evidence-count")
    aware = case("misleading-summary", "A:evidence-count", lineage=True)
    assert naive.value == "rome"
    assert naive.overconfident
    assert aware.value != "rome"
    assert not aware.overconfident


def test_a_conservative_policy_declines_where_an_established_value_meets_one_new_report() -> None:
    for name in ("recent-contradiction", "stale-confidence"):
        assert case(name, "H:conservative").mode == "competing"
        assert case(name, "A:evidence-count").mode == "answer_with_uncertainty"
    assert case("stale-confidence", "A:evidence-count").correct is True
    assert case("recent-contradiction", "A:evidence-count").correct is False


def test_temporal_spoofing_and_reputation_poisoning_succeed_against_most_policies() -> None:
    wrong = [case("temporal-spoofing", p).correct for p in POLICIES]
    assert wrong.count(False) >= 5
    assert all(case("reputation-poisoning", p).correct is False for p in POLICIES)


def test_alternating_evidence_flips_a_naive_policy_more_than_a_corroborating_one() -> None:
    assert case("alternating", "A:evidence-count").flips >= 1
    assert (
        case("alternating", "E:corroboration").flips
        <= case("alternating", "A:evidence-count").flips
    )


def test_identity_uncertainty_lowers_confidence_for_near_collision_entities() -> None:
    plain = case("entity-collision", "A:evidence-count")
    aware = case("entity-collision", "A:evidence-count", identity=True)
    assert plain.score == pytest.approx(0.6)
    assert aware.score == pytest.approx(0.3)
    assert aware.mode == "abstain"
    assert identity_map(["ana.home", "anna.home"]) == {"ana": 0.5, "anna": 0.5}
    assert identity_map(["ana.home", "ben.home"]) == {}


def test_every_case_runs_under_every_policy() -> None:
    results = run_cases()
    # 10 cases, 2 of them also run with lineage awareness and 1 with identity damping
    assert len(results) == 8 * (10 + 2 + 1)
    assert {r.case for r in results} == {c.name for c in adversarial_cases()}
    assert with_signals(POLICIES["A:evidence-count"], "lineage").signal("lineage") == {}


# --- memory operations ---------------------------------------------------------------------------


def test_forgetting_withdraws_evidence_the_ledger_keeps(world: BeliefWorld, tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    c = build_condition(world, "forgetting", store)
    ids = {e.id for e in world.evidence}
    assert c.withdrawals
    assert all(e in ids for _, e in c.withdrawals)
    assert len({t for t, _ in c.withdrawals}) == 1
    assert c.forgetting != "none"
    run = run_world(
        world, POLICIES["B:source-weighted"], evidence=c.evidence, withdrawals=c.withdrawals
    )
    assert {e.id for e in run.ledger.evidence} == ids  # nothing is deleted from the record
    assert violations(run) == []
    assert any(e.kind == "withdraw" for e in run.ledger.events)


def test_consolidation_adds_derived_evidence_that_traces_to_experiences(world: BeliefWorld) -> None:
    c = build_condition(world, "cons-add")
    derived = [e for e in c.evidence if e.status is EpistemicStatus.DERIVED]
    assert derived
    raw_ids = {e.id for e in world.evidence}
    for d in derived:
        assert d.lineage
        assert {x for x, _ in d.lineage} <= raw_ids  # lineage resolves to real experiences
        assert d.multiplicity == len(d.lineage)
    naive = run_world(world, POLICIES["A:evidence-count"], evidence=c.evidence)
    aware = run_world(
        world, with_signals(POLICIES["A:evidence-count"], "lineage"), evidence=c.evidence
    )
    assert violations(naive) == []
    assert violations(aware) == []
    replace = build_condition(world, "cons-replace")
    covered = {x for d in derived for x, _ in d.lineage}
    assert {e for _, e in replace.withdrawals} == covered & raw_ids


def test_interference_adds_reports_and_removes_none(world: BeliefWorld) -> None:
    c = build_condition(world, "interference")
    assert {e.id for e in world.evidence} < {e.id for e in c.evidence}
    assert {
        e.source.partition(":")[0] for e in c.evidence if e.id not in {x.id for x in world.evidence}
    } == {"forum"}
    assert not c.withdrawals


def test_the_operations_study_compares_every_condition_with_the_raw_one(world: BeliefWorld) -> None:
    cals = {p: fit_isotonic(answered_pairs(predictions(run_world(world, POLICIES[p]), world)))
            for p in ("A:evidence-count", "E:corroboration")}  # fmt: skip
    cells = ops_study([world], cals, policies=("A:evidence-count", "E:corroboration"))
    pooled = [c for c in cells if c.world == "ALL"]
    raw = [c for c in pooled if c.condition == "raw"]
    assert {c.policy for c in raw} == {"A:evidence-count", "E:corroboration"}
    assert all(c.d_ece == 0.0 and c.d_correct == 0.0 for c in raw)
    assert {c.condition for c in pooled} == set(CONDITIONS)
    assert any(c.policy.endswith("+lineage") for c in pooled)
    with pytest.raises(ValueError, match="unknown condition"):
        build_condition(world, "nonsense")


def test_belief_and_retrieval_are_compared_on_identical_probes(world: BeliefWorld) -> None:
    preds = predictions(run_world(world, POLICIES["B:source-weighted"]), world)
    r = retrieval_vs_belief(world, preds)
    assert r.both + r.retrieval_only + r.belief_only + r.neither == r.n == len(world.probes)
    assert 0 <= r.p_value <= 1
    assert r.retrieval_ece is not None
    assert r.belief_correct.numerator == sum(bool(p.answered and p.correct) for p in preds)
    with memory_log(world) as log:
        assert len(log.versions()) == len(world.dataset.steps)  # one memory per experience


# --- studies -------------------------------------------------------------------------------------


def test_stability_and_trust_are_measured_from_the_ledger(world: BeliefWorld) -> None:
    run = run_world(world, POLICIES["B:source-weighted"])
    s = stability(run, world, "B")
    assert s.events == len(run.ledger.events)
    assert s.keys == len(world.truth)
    assert s.churn >= s.oscillations >= 0
    t = trust_study(world, run)
    assert t.circular  # learned from the system's own beliefs, and labelled so
    assert t.sources > 0


def test_path_independent_policies_converge_and_history_survives(world: BeliefWorld) -> None:
    for name in ("A:evidence-count", "E:corroboration", "G:provenance-depth"):
        r = order_study(world, POLICIES[name], orders=4, max_delay=15.0)
        assert r.same_final == r.orders, name
        assert r.answer_agreement == 1.0
        assert r.historical_mismatches == 0
    learned = order_study(world, POLICIES["B:source-weighted"], orders=4, max_delay=15.0)
    assert learned.historical_mismatches == 0  # path dependence never breaks reconstruction
    assert reorder(world.evidence, 1, 10.0)[0].occurred_at == world.evidence[0].occurred_at
    assert {e.id for e in reorder(world.evidence, 1, 10.0)} == {e.id for e in world.evidence}


def test_survey_scores_the_classification_against_the_generators_truth(
    world: BeliefWorld, tmp_path: Path
) -> None:
    s, cg = survey(world, ArtifactStore(tmp_path))
    assert s.claims > 0
    assert s.genuine_precision is not None
    assert s.genuine_precision.estimate is not None
    assert (
        s.genuine_precision.estimate > 0.8
    )  # concurrent disagreements involve an erroneous report
    if s.changes_misread is not None:
        assert s.changes_misread.estimate is not None
        assert s.changes_misread.estimate < 0.1  # true changes are rarely read as contradictions
    assert cg.counts == s.relations


def test_the_matrix_fits_recalibration_on_disjoint_seeds_and_compares_pairs(tmp_path: Path) -> None:
    runner = Runner(1, 1)
    mx = matrix(runner, ArtifactStore(tmp_path), world_names=("stable", "contradiction"))
    assert {c.policy for c in mx.pooled} == set(POLICIES)
    assert len(mx.ledgers) == 16
    assert all(c.summary.n == 72 for c in mx.pooled)
    assert dict(mx.calibrators)["A:evidence-count"].fit is not None
    baseline = [c for c in mx.comparisons if c.world == "ALL"]
    assert {c.policy for c in baseline} == set(POLICIES) - {"A:evidence-count"}
    assert all(c.n == 72 for c in baseline)
    stab = next(c for c in baseline if c.policy == "E:corroboration")
    assert stab.correct[3] is not None
    assert {s.tag for s in mx.subgroups} >= {"contradiction-heavy", "evidence:1", "depth:2+"}


# --- provenance ----------------------------------------------------------------------------------


def test_a_belief_traces_to_the_original_experience_or_says_what_is_missing(
    world: BeliefWorld,
) -> None:
    run = run_world(world, POLICIES["B:source-weighted"])
    end = day(world.spec.horizon_days + 30)
    with memory_log(world) as log:
        corpus = Corpus.from_log(log, end)
        hier = consolidate(corpus, HIERARCHICAL, end)
        snap = build_graph(corpus.with_derived(hier.memories, ()), GRAPH_POLICY, [hier])
        probe = next(p for p in world.probes if run.at(p.key, day(p.known_at_day)))
        trace = belief_trace(
            run, corpus, hier, snap, probe.key, day(probe.valid_at_day), day(probe.known_at_day)
        )
        if trace.belief is None:
            pytest.skip("the probe had no answering belief")
        assert trace.complete
        assert trace.missing == ()
        assert trace.arithmetic == "ok"
        assert trace.evidence
        for e in trace.evidence:
            assert e.content is not None
            assert e.content.startswith(("set ", "correct "))
            assert e.memory_versions
            assert e.graph_claim is not None
        assert sum(
            e.weight or 0.0 for e in trace.evidence if e.role == "supporting"
        ) == pytest.approx(
            trace.weight or 0.0, abs=1e-9
        )  # every part of the confidence is reconstructable
        assert trace.events
        early = Corpus.from_log(log, day(-1))  # before any experience
        broken = belief_trace(
            run, early, None, None, probe.key, day(probe.valid_at_day), day(probe.known_at_day)
        )
        assert broken.missing
        assert any("no memory version cites it" in m for m in broken.missing)
        assert not broken.complete


def test_the_whole_study_runs_and_reproduces_in_an_independent_store(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "a")
    spec = study_spec(store, 1, 1, 1, 2, ("delayed-correction",))
    study, perf = run_study(spec, store)
    fresh = ArtifactStore(tmp_path / "b")
    rep = reproduce(study, spec, store, fresh)
    assert rep.identical == rep.ledgers == 8
    assert rep.demonstration_identical
    assert study.demonstration in store.digests("Demonstration")
    assert perf.experiment == study.digest
    demo = store.get_record(Demonstration, study.demonstration)
    assert demo.replay_identical
    assert demo.violations == 0
    assert demo.historical_mismatches == 0
    assert demo.trace.arithmetic == "ok"
    store.verify()
    fresh.verify()
