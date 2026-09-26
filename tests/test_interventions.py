from datetime import timedelta

import pytest

from memoria.core import Dataset, ExpectationStatus, Experience, Step
from memoria.formation import parse_statement
from memoria.interventions import Contaminate, Delay, Drop, Inject, Intervention, Reorder
from memoria.scenarios import (
    UNKNOWN,
    contested,
    day,
    drifting_facts,
    epistemic_truth,
    known,
    relocation_year,
)

BASE = drifting_facts(seed=3, keys=5, changes=8)
ALL: list[Intervention] = [
    Drop(rate=0.3, seed=1),
    Delay(rate=0.5, seed=1, days=15),
    Reorder(seed=1, window_days=20),
    Contaminate(rate=0.5, seed=1, delay_days=2),
    Inject(relocation_year()),
]


def experiences(d: Dataset) -> list[str]:
    return sorted(s.experience.digest for s in d.steps)


@pytest.mark.parametrize("iv", ALL, ids=lambda iv: iv.name)
def test_interventions_are_deterministic_valid_and_keep_probes(iv: Intervention) -> None:
    out = iv.apply(BASE)
    assert out == iv.apply(BASE)
    assert out.probes == BASE.probes  # the instrument is never perturbed
    assert out.version == f"{BASE.version}+{iv.name}"
    assert out.digest != BASE.digest
    times = [s.recorded_at for s in out.steps]
    assert times == sorted(times)  # still a valid ingestion order (Dataset enforces it)
    for s in out.steps:
        assert s.experience.occurred_at <= s.recorded_at  # never ingested before it happened


def test_drop_removes_a_seeded_subset() -> None:
    out = Drop(rate=0.3, seed=1).apply(BASE)
    assert set(experiences(out)) < set(experiences(BASE))
    assert 0 < len(out.steps) < len(BASE.steps)
    assert Drop(rate=0.3, seed=2).apply(BASE) != out


def test_selection_depends_on_content_not_position() -> None:
    subset = Dataset(name=BASE.name, version=BASE.version, steps=BASE.steps[1::2])
    kept_all = {s.digest for s in Drop(rate=0.5, seed=9).apply(BASE).steps}
    kept_subset = {s.digest for s in Drop(rate=0.5, seed=9).apply(subset).steps}
    assert kept_subset == kept_all & {s.digest for s in subset.steps}


@pytest.mark.parametrize("cls", [Drop, Delay, Contaminate])
def test_rate_boundaries(cls: type[Drop | Delay | Contaminate]) -> None:
    kwargs = {"days": 1} if cls is Delay else {}
    assert cls(rate=0, seed=1, **kwargs).apply(BASE).steps == BASE.steps  # type: ignore[arg-type]
    everything = cls(rate=1, seed=1, **kwargs).apply(BASE)  # type: ignore[arg-type]
    if cls is Drop:
        assert everything.steps == ()
    elif cls is Delay:
        assert all(
            a.recorded_at + timedelta(days=1) == b.recorded_at
            for a, b in zip(BASE.steps, everything.steps, strict=True)
        )
    else:
        assert len(everything.steps) == 2 * len(BASE.steps)


def test_delay_moves_record_time_only() -> None:
    out = Delay(rate=1, seed=1, days=15).apply(BASE)
    assert experiences(out) == experiences(BASE)
    for a, b in zip(BASE.steps, out.steps, strict=True):
        assert b.experience == a.experience
        assert b.recorded_at - a.recorded_at == timedelta(days=15)


def test_reorder_never_moves_ingestion_earlier() -> None:
    out = Reorder(seed=4, window_days=30).apply(BASE)
    assert experiences(out) == experiences(BASE)
    planned = {s.experience.digest: s.recorded_at for s in BASE.steps}
    assert all(s.recorded_at >= planned[s.experience.digest] for s in out.steps)
    order = [s.experience.digest for s in out.steps]
    assert order != [s.experience.digest for s in BASE.steps]  # something did move


def test_contaminants_are_traceable_to_what_they_imitate() -> None:
    out = Contaminate(rate=1, seed=1, delay_days=2).apply(BASE)
    originals = {s.experience.digest: s for s in BASE.steps}
    fakes = [s for s in out.steps if s.experience.source.startswith("contaminant:")]
    assert len(fakes) == len(BASE.steps)
    for fake in fakes:
        original = originals[fake.experience.source.removeprefix("contaminant:")]
        f, o = (
            parse_statement(fake.experience.content),
            parse_statement(original.experience.content),
        )
        assert f is not None
        assert o is not None
        assert (f.verb, f.key, f.value) == (o.verb, o.key, f"{o.value} (contaminated)")
        assert fake.experience.occurred_at - original.experience.occurred_at == timedelta(days=2)
        assert fake.recorded_at - original.recorded_at == timedelta(days=2)


def test_contaminate_ignores_non_statements() -> None:
    out = Contaminate(rate=1, seed=1).apply(relocation_year())
    fakes = [s.experience.content for s in out.steps if "contaminated" in s.experience.content]
    # set/correct statements only: not chatter, not "forget", not malformed statements.
    assert len(fakes) == 12
    assert all(c.startswith(("set ", "correct ")) for c in fakes)


def test_inject_merges_originals_first_on_ties() -> None:
    extra = Step(
        experience=Experience(source="x", content="set k0 = injected", occurred_at=day(0)),
        recorded_at=BASE.steps[0].recorded_at,
    )
    source = Dataset(name="extra", version="1", steps=(extra,))
    out = Inject(source).apply(BASE)
    assert out.steps.index(extra) == 1
    assert Inject(source).params == (("dataset", source.digest),)


def test_interventions_compose() -> None:
    once = Delay(rate=0.5, seed=1, days=5).apply(Drop(rate=0.2, seed=1).apply(BASE))
    assert once.version.endswith("+drop+delay")
    assert once == Delay(rate=0.5, seed=1, days=5).apply(Drop(rate=0.2, seed=1).apply(BASE))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Drop(rate=1.5, seed=1),
        lambda: Drop(rate=-0.1, seed=1),
        lambda: Drop(rate=0.5, seed=True),
        lambda: Delay(rate=0.5, seed=1, days=0),
        lambda: Reorder(seed=1, window_days=0),
        lambda: Contaminate(rate=0.5, seed=1, delay_days=-1),
        lambda: Contaminate(rate=0.5, seed=1, source=""),
        lambda: Contaminate(rate=0.5, seed=1, source="a:b"),
    ],
)
def test_invalid_parameters_rejected(build: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        build()  # type: ignore[operator]


def test_empty_dataset() -> None:
    empty = Dataset(name="e", version="1", steps=())
    for iv in ALL:
        assert iv.apply(empty).steps == (() if iv.name != "inject" else relocation_year().steps)


# --- datasets and ground truth ---------------------------------------------------------------


def test_relocation_year_is_a_valid_fixed_dataset() -> None:
    d = relocation_year()
    assert d == relocation_year()
    assert len(d.steps) == 16
    assert len(d.probes) == 12
    statuses = {p.expected.status for p in d.probes}
    assert statuses == set(ExpectationStatus)


def test_drifting_facts_is_seeded_and_well_formed() -> None:
    a = drifting_facts(seed=1)
    assert a == drifting_facts(seed=1)
    assert a != drifting_facts(seed=2)
    assert len(a.steps) == 4 * 5
    assert len(a.probes) == 4 * 6
    assert all(s.experience.occurred_at <= s.recorded_at for s in a.steps)
    assert "seed=1" in a.version
    # Pinned: the generator is a function of its parameters on every platform.
    assert a.digest == "sha256:47248907aebc0519a34dd437ee1d157f992793483c00cd1c4840f5013fb7859d"


def test_drifting_facts_rejects_bad_parameters() -> None:
    for kwargs in ({"keys": 0}, {"changes": 0}, {"horizon_days": 0}, {"max_delay_days": -1}):
        with pytest.raises(ValueError, match="horizon"):
            drifting_facts(seed=1, **kwargs)


def test_drifting_facts_probes_include_not_yet_known() -> None:
    d = drifting_facts(seed=5, keys=6, changes=6, probes_per_key=20, max_delay_days=30)
    statuses = [p.expected.status for p in d.probes]
    assert ExpectationStatus.KNOWN in statuses
    assert ExpectationStatus.UNKNOWN in statuses


def test_epistemic_truth_rules() -> None:
    r = [
        (day(10), day(12), "a"),  # occurred 10, known from 12
        (day(20), day(40), "b"),  # occurred 20, reported late (40)
        (day(15), day(16), "c"),  # occurred 15
    ]
    assert epistemic_truth(r, valid_at=day(5), known_at=day(100)) == UNKNOWN  # before all
    assert epistemic_truth(r, valid_at=day(11), known_at=day(11)) == UNKNOWN  # not yet known
    assert epistemic_truth(r, valid_at=day(11), known_at=day(12)) == known("a")
    assert epistemic_truth(r, valid_at=day(25), known_at=day(30)) == known("c")  # b unknown yet
    assert epistemic_truth(r, valid_at=day(25), known_at=day(40)) == known("b")
    # A later-ingested report is not more true: occurrence order decides.
    late_old = [*r, (day(12), day(90), "d")]
    assert epistemic_truth(late_old, valid_at=day(25), known_at=day(95)) == known("b")
    tie = [(day(10), day(10), "x"), (day(10), day(11), "y")]
    assert epistemic_truth(tie, valid_at=day(10), known_at=day(11)) == contested("x", "y")
    same = [(day(10), day(10), "x"), (day(10), day(11), "x")]
    assert epistemic_truth(same, valid_at=day(10), known_at=day(11)) == known("x")
    assert epistemic_truth([], valid_at=day(0), known_at=day(0)) == UNKNOWN
