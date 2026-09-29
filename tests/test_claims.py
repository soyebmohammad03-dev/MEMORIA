import pytest

from memoria.adaptive_cases import (
    EXTRACTION_CASES,
    IDENTITY_CASES,
    ExtractionCase,
    IdentityCase,
    merge_cases,
    run_identity_cases,
)
from memoria.adaptive_worlds import registry_for
from memoria.claims import EXTRACTOR, extract, text_digest, verify_claim
from memoria.identity import (
    CONSERVATIVE,
    EXACT_ONLY,
    SIMILARITY,
    IdentityAssertion,
    IdentityPolicy,
    false_merges,
    make_registry,
    merge_decisions,
    resolve_mention,
)

REG = registry_for(True)


@pytest.mark.parametrize("case", EXTRACTION_CASES, ids=lambda c: c.name)
def test_designed_extraction_cases(case: ExtractionCase) -> None:
    got = tuple(
        (c.status, c.reason, c.entity, c.attribute, c.value, c.kind)
        for c in extract(case.text, "e", REG)
    )
    assert got == case.expected


def test_extracted_claims_reproduce_their_source_spans() -> None:
    text = "Ana lives in Berlin. Ben works at Acme Corp."
    claims = extract(text, "e1", REG)
    assert [c.value for c in claims] == ["berlin", "acme corp"]
    for c in claims:
        assert verify_claim(c, text) == []
        assert c.extractor == EXTRACTOR.digest
        assert c.experience == "e1"
        assert c.text == text_digest(text)
    assert claims[0].value_span is not None
    s, e = claims[0].value_span
    assert text[s:e] == "Berlin"


def test_lineage_verification_detects_a_different_text() -> None:
    (claim,) = extract("Ana lives in Berlin.", "e", REG)
    assert verify_claim(claim, "Ana lives in Berlin.") == []
    assert "text digest differs" in verify_claim(claim, "Ana lives in Rome.")


def test_unresolved_claims_keep_the_proposal_but_expose_no_fact() -> None:
    (claim,) = extract("I think Ana lives in Rome.", "e", REG)
    assert claim.status == "unresolved"
    assert claim.reason == "hedged"
    assert (claim.entity, claim.attribute, claim.value, claim.kind) == (None, None, None, None)
    assert (claim.proposed_subject, claim.proposed_value, claim.proposed_kind) == (
        "Ana",
        "Rome",
        "assert",
    )
    assert claim.key is None
    assert claim.confidence == 0
    assert verify_claim(claim, "I think Ana lives in Rome.") == []


def test_extraction_is_deterministic_and_ids_are_content_derived() -> None:
    a = extract("Ana lives in Rome. She works at Acme.", "e", REG)
    assert a == extract("Ana lives in Rome. She works at Acme.", "e", REG)
    assert len({c.id for c in a}) == 2
    assert extract("Ana lives in Rome.", "other", REG)[0].id != a[0].id


def test_ambiguity_is_never_resolved_by_picking_one_entity() -> None:
    (claim,) = extract("Ann lives in Rome.", "e", REG)
    assert claim.reason == "ambiguous_entity"
    assert claim.candidates == ("ana", "anna")


@pytest.mark.parametrize("case", IDENTITY_CASES, ids=lambda c: c.name)
def test_identity_cases_per_policy(case: IdentityCase) -> None:
    for policy in (CONSERVATIVE, EXACT_ONLY, SIMILARITY):
        r = resolve_mention(case.mention, REG, policy)
        assert f"{r.outcome}:{r.entity or '-'}" == case.expected[policy.name]


def test_similarity_alone_never_establishes_identity_when_conservative() -> None:
    for name in ("Aana", "Annaa", "Benno", "Ann", "Ana K.", "Bennn"):
        r = resolve_mention(name, REG, CONSERVATIVE)
        assert r.outcome != "resolved" or r.rule in ("exact", "alias")


def test_false_merges_are_detectable_and_only_the_similarity_policy_makes_them() -> None:
    rows = run_identity_cases()
    assert all(passed for _, _, passed, _, _ in rows)
    false = {(n, p) for n, p, _, _, fm in rows if fm}
    assert false == {("unregistered-near-name", "similarity")}


def test_identity_assertions_need_independent_sources_and_no_declared_distinction() -> None:
    by = {name: d for name, d, _ in merge_cases()}
    assert by["two-independent-sources-same-person"].outcome == "merged"
    assert by["one-source-only"].outcome == "insufficient"
    assert by["declared-distinct-conflict"].outcome == "conflict"
    flags = {name: fm for name, _, fm in merge_cases()}
    assert flags["undeclared-false-merge"]
    assert not flags["two-independent-sources-same-person"]


def test_a_merge_chain_cannot_join_declared_distinct_entities() -> None:
    reg = make_registry({"a": ("A", []), "b": ("B", []), "c": ("C", [])}, distinct=[("a", "c")])
    xs = [
        IdentityAssertion(a=x, b=y, source=s, evidence="e")
        for x, y in (("a", "b"), ("b", "c"))
        for s in ("s1", "s2")
    ]
    out = merge_decisions(reg, xs, IdentityPolicy(name="p"))
    assert [d.outcome for d in out] == ["merged", "conflict"]
    assert false_merges(out, {"a": "1", "b": "1", "c": "3"}) == (0, 1)
