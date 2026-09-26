"""End-to-end: experiences -> formation -> state -> retrieval -> response -> provenance.

Phase 2 exit criterion: a response is traceable to exact version digests.
"""

from pathlib import Path

import pytest

from memoria.core import (
    FormationDecision,
    Query,
    Response,
    RetrievalTrace,
    check_response,
    check_trace,
    content_hash,
)
from memoria.formation import EpisodicPolicy, FormationPolicy, StatementPolicy, form
from memoria.retrieval import answer, lexical_recency
from memoria.scenarios import day, relocation_year
from memoria.store import MemoryLog

QUESTIONS = [
    ("where is home", 57, 59),
    ("where is home", 57, 61),
    ("where is home", 345, 345),
    ("who is my employer", 10, 60),
    ("who is my employer", 10, 70),
    ("who is my employer", 175, 175),
    ("pet", 345, 345),
    ("zebra", 345, 345),
]


def build(path: Path, policy: FormationPolicy) -> list[str]:
    """Form the scenario, ask every question; return all record digests in order."""
    with MemoryLog(path) as log:
        for step in relocation_year().steps:
            form(log, step.experience, policy, recorded_at=step.recorded_at)
        for text, valid, known in QUESTIONS:
            q = Query(text=text, valid_at=day(valid), known_at=day(known), limit=3)
            answer(log, lexical_recency(), q)
        return (
            [d.digest for d in log.records(FormationDecision)]
            + [t.digest for t in log.records(RetrievalTrace)]
            + [r.digest for r in log.records(Response)]
        )


@pytest.mark.parametrize("policy", [StatementPolicy(), EpisodicPolicy()], ids=lambda p: p.name)
def test_every_response_is_traceable_to_exact_versions(
    tmp_path: Path, policy: FormationPolicy
) -> None:
    build(tmp_path / "log.db", policy)
    with MemoryLog(tmp_path / "log.db") as log:  # reopened: persistence, not memory
        log.verify()
        decisions = {d.version: d for d in log.records(FormationDecision) if d.version}
        responses = log.records(Response)
        assert len(responses) == len(QUESTIONS)
        for response in responses:
            trace = log.record(RetrievalTrace, response.trace)
            assert trace is not None
            check_response(response, trace)
            q = trace.query
            check_trace(trace, log.state_as_of(valid_at=q.valid_at, known_at=q.known_at))
            for digest in response.cited:
                version = log.version(digest)  # exact version, by content address
                assert version is not None
                assert version.content == response.output
                decision = decisions[digest]  # the recorded operation that produced it
                assert decision.operation is version.operation
                for source in version.derived_from:  # the original experiences
                    experience = log.experience(source)
                    assert experience is not None
                    assert experience.occurred_at <= version.recorded_at
                if version.supersedes:  # and the history it replaced
                    assert log.version(version.supersedes) is not None


def test_abstention_is_recorded_with_its_evidence(tmp_path: Path) -> None:
    build(tmp_path / "log.db", StatementPolicy())
    with MemoryLog(tmp_path / "log.db") as log:
        by_text = {(r.query.text, r.query.valid_at): r for r in log.records(RetrievalTrace)}
        zebra = by_text[("zebra", day(345))]
        assert zebra.selected == ()
        assert len(zebra.candidates) == 3  # every held memory was considered and rejected
        (response,) = [r for r in log.records(Response) if r.trace == zebra.digest]
        assert response.output is None


def test_whole_pipeline_is_reproducible(tmp_path: Path) -> None:
    """Independent runs produce byte-identical decisions, traces and responses."""
    for policy in (StatementPolicy(), EpisodicPolicy()):
        first = build(tmp_path / f"{policy.name}-a.db", policy)
        second = build(tmp_path / f"{policy.name}-b.db", policy)
        assert first == second


def test_stored_traces_are_reproducible_from_the_log(tmp_path: Path) -> None:
    build(tmp_path / "log.db", StatementPolicy())
    with MemoryLog(tmp_path / "log.db") as log:
        for trace in log.records(RetrievalTrace):
            q = trace.query
            state = log.state_as_of(valid_at=q.valid_at, known_at=q.known_at)
            assert lexical_recency().retrieve(state, q) == trace


# Pinned digests of every decision, trace and response the scenario produces. A change
# here means stored results would differ: either a deliberate semantic change (update
# the pin and record it in the decision log) or platform-dependent drift (a bug).
GOLDEN = {
    "statement-v1": "sha256:231c30c575b2bef568edddf495469dd2f3d9df3e3855bb76b98c74028ec5d09f",
    "episodic-v1": "sha256:18a78969af9935fe3dbdbcc7e24d3a5915b7e8de548960fe8b16db1a2123db28",
}


@pytest.mark.parametrize("policy", [StatementPolicy(), EpisodicPolicy()], ids=lambda p: p.name)
def test_pipeline_matches_golden_digest(tmp_path: Path, policy: FormationPolicy) -> None:
    assert content_hash(build(tmp_path / "log.db", policy)) == GOLDEN[policy.name]
