import pytest
from pydantic import ValidationError

from memoria.entities import (
    AGGRESSIVE,
    CONSERVATIVE,
    ResolutionPolicy,
    false_merge_rate,
    initials_match,
    normalize_name,
    resolve,
    surface_names,
    trigram_jaccard,
)

X = "sha256:" + "1" * 64
Y = "sha256:" + "2" * 64
Z = "sha256:" + "3" * 64


def test_surface_names_are_capitalised_runs_without_sentence_function_words() -> None:
    text = "Ana's home is Paris. The Alex K. met Alexander Kumar. It rained."
    assert surface_names(text) == ["Ana", "Paris", "Alex K.", "Alexander Kumar"]
    assert normalize_name("Alex K.") == "alex k"
    assert normalize_name("ANA's") == "ana"


def test_similar_names_are_candidates_never_merged_by_default() -> None:
    r = resolve({}, {"Alex Kumar": [X], "Alex K.": [Y], "Alexander Kumar": [Z]}, CONSERVATIVE)
    pairs = {(d.a, d.b): d for d in r.decisions}
    d = pairs[("surface:alex k", "surface:alex kumar")]
    assert d.rule == "similar"
    assert not d.accepted
    assert d.reason == "candidate_only"
    assert d.evidence == (X, Y)
    assert len(r.entities) == 3  # three unresolved entities, not one
    assert set(r.unresolved) == {e for e, _ in r.entities}


def test_identity_rules_merge_with_declared_confidence() -> None:
    r = resolve({"ana": [X]}, {"Ana": [Y], "ANA": [Z]}, CONSERVATIVE)
    assert r.entities == (("entity:ana", ("structured:ana", "surface:ana")),)
    (d,) = r.decisions
    assert (d.rule, d.accepted, d.confidence) == ("normalized", True, 0.9)
    assert r.entity_of("surface:ana") == "entity:ana"
    assert r.false_merges == ()


def test_alias_and_reversal() -> None:
    r = resolve({"ana": [X]}, {"Boss Ana": [Y]}, CONSERVATIVE)
    assert r.entity_of("surface:boss ana") == "entity:~boss ana"  # no alias: unresolved
    r = resolve({"ana": [X]}, {"Boss Ana": [Y]}, ResolutionPolicy(
        name="a", version="1", aliases=(("boss ana", "ana"),)))  # fmt: skip
    assert r.entity_of("surface:boss ana") == "entity:ana"
    reverted = ResolutionPolicy(name="a", version="2", aliases=(("boss ana", "ana"),),
                                rejected=(("structured:ana", "surface:boss ana"),))  # fmt: skip
    r2 = resolve({"ana": [X]}, {"Boss Ana": [Y]}, reverted)
    assert r2.entity_of("surface:boss ana") != "entity:ana"
    assert any(d.reason == "rejected_by_policy" for d in r2.decisions)


def test_entity_collision_is_a_reported_false_merge_under_aggressive_resolution() -> None:
    structured = {"ana": [X], "anna": [Y]}
    assert trigram_jaccard("ana", "anna") >= AGGRESSIVE.threshold
    safe = resolve(structured, {}, CONSERVATIVE)
    assert safe.false_merges == ()
    risky = resolve(structured, {}, AGGRESSIVE)
    assert risky.false_merges == ("entity:ana",)
    truth = {"structured:ana": "ana", "structured:anna": "anna"}
    assert false_merge_rate(risky, truth) == (1, 1)
    assert false_merge_rate(safe, truth) == (0, 0)


def test_policies_are_closed_and_canonical() -> None:
    with pytest.raises(ValidationError):
        ResolutionPolicy(name="x", version="1", accept=("structured", "alias"))  # unsorted
    with pytest.raises(ValidationError):
        ResolutionPolicy(name="x", version="1", rejected=(("b", "a"),))
    assert initials_match("alex k", "alex kumar")
    assert not initials_match("alex", "alexander")
    assert CONSERVATIVE.digest != AGGRESSIVE.digest
