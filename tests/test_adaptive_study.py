import pytest

from memoria.adaptive_study import (
    ABLATIONS,
    COMPARISONS,
    AdaptiveStudy,
    StudySpec,
    reproduce,
    run_study,
    study_spec,
    systems,
)
from memoria.artifacts import ArtifactStore


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> tuple[AdaptiveStudy, ArtifactStore]:
    store = ArtifactStore(tmp_path_factory.mktemp("s1"))
    spec = study_spec(store, replicates=2, cal_replicates=1, world_names=("changing",))
    result, _ = run_study(spec, store)
    return result, store


def test_every_comparison_names_known_systems() -> None:
    table = systems()
    for t, b in COMPARISONS:
        assert t in table
        assert b in table
    assert all(a in table for a in ABLATIONS)
    assert len({s.digest for s in table.values()}) == len(table)


def test_the_study_reproduces_from_its_spec_alone(
    study: tuple[AdaptiveStudy, ArtifactStore], tmp_path_factory: pytest.TempPathFactory
) -> None:
    result, store = study
    spec = store.get_record(StudySpec, result.spec)
    rep = reproduce(result, spec, ArtifactStore(tmp_path_factory.mktemp("fresh")), [0, 1])
    assert rep.differing == ()
    assert rep.identical == rep.runs_checked == 2 * len(systems())


def test_a_second_execution_gives_the_same_study_digest(
    study: tuple[AdaptiveStudy, ArtifactStore], tmp_path_factory: pytest.TempPathFactory
) -> None:
    other = ArtifactStore(tmp_path_factory.mktemp("s2"))
    spec = study_spec(other, replicates=2, cal_replicates=1, world_names=("changing",))
    again, _ = run_study(spec, other)
    assert again.digest == study[0].digest


def test_paired_comparisons_are_adjusted_and_labelled(
    study: tuple[AdaptiveStudy, ArtifactStore],
) -> None:
    result, _ = study
    assert result.comparisons
    for c in result.comparisons:
        assert c.holm_p >= c.diff.sign_p - 1e-12  # Holm never lowers a p-value
        assert c.verdict in ("worse", "better", "no_detected_difference", "underpowered")
        if c.verdict in ("worse", "better"):
            assert c.holm_p < 0.05
            assert not c.diff.underpowered


def test_summaries_are_consistent_with_their_counts(
    study: tuple[AdaptiveStudy, ArtifactStore],
) -> None:
    result, store = study
    assert {s.system for s in result.summaries} == set(systems())
    for s in result.summaries:
        m = {n: p for n, p, _ in s.metrics}
        parts = [m[k] for k in ("correct_rate", "stale_rate", "wrong_rate", "none_rate")]
        assert all(p is not None for p in parts)
        counted = sum(p.numerator for p in parts if p is not None)
        assert parts[0] is not None
        assert counted == parts[0].denominator  # every probe is exactly one of the four outcomes
    assert len(result.runs) == 2 * len(systems())
    for _, _, _, digest in result.runs:
        assert store.has(digest)


def test_onsets_and_doses_report_a_grid_point_or_none(
    study: tuple[AdaptiveStudy, ArtifactStore],
) -> None:
    result, _ = study
    for o in result.onsets:
        assert o.onset is None or o.onset in [h for h, _, _ in o.points]
    for d in result.dose_onsets:
        assert d.onset is None or d.onset in [x for x, *_ in d.doses]


def test_traces_have_no_violations_and_calibration_is_fitted_apart(
    study: tuple[AdaptiveStudy, ArtifactStore],
) -> None:
    result, _ = study
    for a in result.audits:
        assert (
            a.cutoff_violations == a.reproduction_violations == a.unresolved_importance_links == 0
        )
    for c in result.calibrations:
        assert c.n_fit > 0
        assert c.auroc_use is not None
        assert c.raw_ece is not None
        assert c.recalibrated_ece is not None
        assert c.recalibrated_ece <= c.raw_ece  # the raw importance is a ranking, not a probability
