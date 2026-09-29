from typing import Any

import pytest
from pydantic import ValidationError

from memoria.consolidation_eval import RETRIEVAL, STRATEGIES
from memoria.embeddings import HashedNgramEmbedder
from memoria.entities import AGGRESSIVE, CONSERVATIVE
from memoria.interference import (
    MECHANISMS,
    InterferenceSpec,
    Mechanism,
    Observation,
    Role,
    false_merges,
    load_curve,
    manipulation,
    run_scenario,
    scenario,
)
from memoria.memory_lab import POLICIES
from memoria.statistics import paired_mean, t_cdf, t_quantile

EMB = HashedNgramEmbedder()


def spec(mechanism: Mechanism, load: int, replicate: int = 0, **kw: Any) -> InterferenceSpec:
    wording = "paraphrase" if mechanism == "consolidation" else "template"
    return InterferenceSpec.model_validate(
        {"name": "t", "seed": 5, "mechanism": mechanism, "replicate": replicate, "load": load,
         "wording": wording} | kw
    )  # fmt: skip


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_populations_are_deterministic_nested_and_fully_labelled(mechanism: Mechanism) -> None:
    _, sc_small = scenario(spec(mechanism, 4))
    big, sc_big = scenario(spec(mechanism, 8))
    assert scenario(spec(mechanism, 8)) == (big, sc_big)
    assert {x.source for x in sc_small.labels} < {x.source for x in sc_big.labels}
    assert {s.experience.source for s in big.steps} == {x.source for x in sc_big.labels}
    (target,) = [x for x in sc_big.labels if x.role == "target"]
    assert (target.key, target.value) == (sc_big.target_key, sc_big.target_value)
    (probe,) = big.probes
    # exact ground truth (the scenario stores it normalised, as the comparator reads it)
    assert [v.casefold() for v in probe.expected.values] == [sc_big.target_value]
    assert probe.key == sc_big.target_key
    assert len([x for x in sc_big.labels if x.role == "distractor"]) == 8
    _, sc0 = scenario(spec(mechanism, 0))
    assert [x.role for x in sc0.labels] == ["target"]


def test_each_mechanism_moves_its_target_property() -> None:
    def checks(mechanism: Mechanism, **kw: Any) -> dict[str, float]:
        ds, sc = scenario(spec(mechanism, 16, **kw))
        return manipulation(sc, ds, EMB)

    assert checks("proactive")["same_key"] == 1.0
    assert checks("proactive")["newer_than_target"] == 0.0
    assert checks("retroactive")["after_valid_time"] == 1.0
    assert checks("contradiction")["same_instant"] == 1.0
    assert checks("entity")["same_entity"] == 1.0
    assert checks("entity")["same_key"] == 0.0
    assert checks("semantic")["same_entity"] == 0.0
    # wording moves similarity to the query: paraphrases share its words ("lives")
    assert (
        checks("semantic", wording="paraphrase")["mean_query_cosine"]
        > checks("semantic")["mean_query_cosine"]
    )
    assert checks("semantic", recency="newer")["newer_than_target"] == 1.0
    assert checks("contradiction", frequency=3)["copies"] == 3.0
    temporal = checks("temporal")
    assert 0 < temporal["newer_than_target"] < 1


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        InterferenceSpec(name="t", seed=1, mechanism="consolidation", replicate=0, load=1)
    with pytest.raises(ValidationError):
        InterferenceSpec(name="t", seed=1, mechanism="semantic", replicate=0, load=-1)


def run(mechanism: Mechanism, load: int, policy: str, replicate: int = 0) -> Observation:
    ds, sc = scenario(spec(mechanism, load, replicate))
    (o,) = run_scenario(ds, sc, [(policy, POLICIES[policy])], EMB, STRATEGIES["D:abstractive"])
    return o


def test_load_zero_is_the_controlled_baseline() -> None:
    mechanisms: tuple[Mechanism, ...] = ("proactive", "contradiction", "semantic", "entity")
    for m in mechanisms:
        o = run(m, 0, "mixed")
        assert o.hit
        assert o.top[0] == "target"
        assert not o.temporal_confused
        assert not o.entity_confused
        assert not o.provenance_confused


def test_temporal_filter_blocks_retroactive_interference() -> None:
    kept = run("retroactive", 4, "mixed")
    assert kept.hit
    lost = run("retroactive", 4, "mixed-no-temporal")
    assert not lost.hit
    assert lost.temporal_confused
    assert lost.top[0] == "distractor"


def test_contradiction_interference_exposes_rivals() -> None:
    o = run("contradiction", 8, "mixed")
    assert o.rival_values_in_top > 0


def test_consolidation_derived_memories_are_roles_not_evidence() -> None:
    o = run("consolidation", 8, "mixed")
    assert set(o.top) <= {"target", "target-derived", "distractor", "distractor-derived"}


def test_near_collisions_are_false_merges_only_under_aggressive_resolution() -> None:
    ds, sc = scenario(spec("semantic", 16, wording="paraphrase"))
    wrong, merged = false_merges(ds, sc, CONSERVATIVE)
    assert wrong == 0
    wrong_a, merged_a = false_merges(ds, sc, AGGRESSIVE)
    assert merged_a >= merged
    assert wrong_a >= wrong
    # one-syllable differences in six-letter names stay below the trigram threshold (0.4 <
    # 0.5): a measured property of the aggressive policy, reported by the lab
    assert (wrong_a, merged_a) == (0, 0)


def observation(rep: int, load: int, rank: int | None, top: Role = "target") -> Observation:
    return Observation(
        mechanism="proactive",
        policy="p",
        replicate=rep,
        load=load,
        trace="sha256:" + "0" * 64,
        corpus="sha256:" + "0" * 64,
        target_rank=rank,
        source_rank=rank,
        target_excluded=False,
        top=(top,),
        top1_entity=None,
        top1_key=None,
        top1_value=None,
        rival_values_in_top=0,
        entropy=None,
        derived_merges_across=0,
        temporal_confused=False,
        entity_confused=False,
        provenance_confused=False,
    )


def test_load_curve_separates_interference_from_retrieval_failure() -> None:
    obs = []
    for r in range(12):
        base_hit = r < 10  # replicates 10, 11 already fail at load 0: not interference
        obs.append(observation(r, 0, 1 if base_hit else 2, "target" if base_hit else "other"))
        displaced = r < 8
        obs.append(observation(r, 4, 3 if displaced else (1 if base_hit else 2),
                               "distractor" if displaced else ("target" if base_hit
                                                               else "other")))  # fmt: skip
    curve = load_curve(obs, "proactive", "p")
    zero, four = curve.points
    assert zero.regime == "baseline"
    assert (four.baseline_failures, four.interference_failures, four.other_failures) == (2, 8, 0)
    assert four.target_at_1.numerator == 2
    assert four.regime == "degraded"
    assert curve.onset == 4
    assert four.rank_displacement.mean is not None
    assert four.rank_displacement.mean > 0
    assert four.rr_degradation.positive == 8
    assert four.difference is not None
    assert four.difference[0] < 0


def test_student_t_against_reference_values() -> None:
    for df, ref in ((1, 12.706204736174707), (5, 2.5705818366147395), (10, 2.2281388519649385),
                    (30, 2.0422724563012373)):  # fmt: skip
        assert t_quantile(0.975, df) == pytest.approx(ref, abs=1e-9)
    assert t_cdf(0.0, 7) == pytest.approx(0.5)
    m = paired_mean([1.0, 2.0, 3.0, 0.0, -1.0, 2.0], 0.95)
    assert m.mean == pytest.approx(7 / 6)
    assert m.low is not None
    assert m.high is not None
    assert m.mean is not None
    assert m.low < m.mean < m.high
    assert (m.positive, m.negative) == (4, 1)
    assert m.underpowered
    assert paired_mean([], 0.95).mean is None
    same = paired_mean([0.5, 0.5, 0.5], 0.95)
    assert same.d_z is None
    assert same.low == same.high == 0.5


def test_mixed_and_raw_policies_are_both_available() -> None:
    assert RETRIEVAL["mixed"].digest == POLICIES["mixed"].digest
