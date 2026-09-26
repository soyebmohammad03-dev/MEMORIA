import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

import pytest

from memoria.core import (
    MemoryState,
    MemoryVersion,
    Operation,
    Query,
    RetrievalTrace,
    Scalar,
    SignalEvidence,
    quantize,
    state_as_of,
)
from memoria.formation import EpisodicPolicy, StatementPolicy, form
from memoria.retrieval import (
    BM25,
    EXTRACTIVE,
    Recency,
    Retriever,
    answer,
    extractive,
    lexical,
    lexical_recency,
    tokenize,
)
from memoria.scenarios import day, relocation_year
from memoria.store import MemoryLog

DIGEST = "sha256:" + "0" * 64


def mem(memory_id: str, content: str, at: float = 0) -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        version=1,
        operation=Operation.CREATE,
        content=content,
        valid_from=day(at),
        recorded_at=day(at),
        derived_from=(DIGEST,),
    )


def state(*memories: MemoryVersion, at: float = 10) -> MemoryState:
    return state_as_of(memories, valid_at=day(at), known_at=day(at))


def query(text: str, at: float = 10, limit: int = 5) -> Query:
    return Query(text=text, valid_at=day(at), known_at=day(at), limit=limit)


CORPUS = (
    mem("home", "home = Paris"),
    mem("pet", "pet = dog"),
    mem("tea", "drink = tea and home cooking"),
)


def test_tokenize() -> None:
    assert tokenize("Home = PARIS, Straße!") == ["home", "paris", "strasse"]  # casefold
    assert tokenize("  ") == []
    assert tokenize("ÉCOLE école") == ["école", "école"]


def test_bm25_matches_hand_computation() -> None:
    (home, pet, tea) = BM25().score(query("home"), CORPUS)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))

    def expected(doc_len: int) -> float:
        return idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * doc_len / 3))

    assert home.score == quantize(expected(2))
    assert tea.score == quantize(expected(5))
    assert pet.score == 0
    inputs = dict(home.inputs)
    assert (inputs["tf:home"], inputs["df:home"], inputs["doc_len"], inputs["n_docs"]) == (
        1,
        2,
        2,
        3,
    )
    assert inputs["idf:home"] == quantize(idf)
    assert inputs["contribution:home"] == home.score
    assert "tf:home" not in dict(pet.inputs)


def test_bm25_rewards_rarer_terms_and_dedups_query() -> None:
    once = BM25().score(query("home paris"), CORPUS)
    twice = BM25().score(query("home home paris PARIS"), CORPUS)
    assert once == twice
    idf = dict(once[0].inputs)
    assert float(idf["idf:paris"]) > float(idf["idf:home"])


@pytest.mark.parametrize("params", [{"k1": -1}, {"b": 1.5}, {"b": -0.1}])
def test_bm25_rejects_bad_params(params: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="BM25"):
        BM25(**params)


def test_recency_half_life_and_axes() -> None:
    late = MemoryVersion.model_validate(mem("m", "x", at=0).model_dump() | {"recorded_at": day(20)})
    q = Query(text="x", valid_at=day(30), known_at=day(50), limit=1)
    (by_record,) = Recency(half_life_days=30).score(q, [late])
    (by_valid,) = Recency(half_life_days=30, axis="valid").score(q, [late])
    assert (by_record.score, dict(by_record.inputs)["age_days"]) == (0.5, 30)
    assert (by_valid.score, dict(by_valid.inputs)["age_days"]) == (0.5, 30)
    (fresh,) = Recency().score(query("x", at=0), [mem("m", "x")])
    assert fresh.score == 1.0


@pytest.mark.parametrize("params", [{"half_life_days": 0}, {"axis": "wall"}])
def test_recency_rejects_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Recency"):
        Recency(**params)  # type: ignore[arg-type]


# --- retriever ---------------------------------------------------------------------------------


def test_every_memory_is_a_candidate_with_full_evidence() -> None:
    trace = lexical_recency().retrieve(state(*CORPUS), query("home", limit=1))
    assert {c.memory_id for c in trace.candidates} == {"home", "pet", "tea"}
    assert [c.memory_id for c in trace.candidates] == ["home", "tea", "pet"]
    assert [c.eligible for c in trace.candidates] == [True, True, False]
    assert trace.selected == (CORPUS[0].digest,)
    for c in trace.candidates:
        assert [s.signal for s in c.signals] == ["bm25", "recency"]
    # Ineligible memories still carry recency evidence: why-not is answerable.
    assert trace.candidates[-1].total > 0
    assert trace.state == state(*CORPUS).digest


def test_spec_records_every_parameter() -> None:
    spec = lexical_recency(half_life_days=7, recency_weight=0.25).spec
    assert spec.gate == "bm25"
    assert [(s.name, s.weight, dict(s.params)) for s in spec.signals] == [
        ("bm25", 1.0, {"b": 0.75, "k1": 1.2}),
        ("recency", 0.25, {"axis": "recorded", "half_life_days": 7}),
    ]


def test_ties_break_by_memory_id() -> None:
    twins = (mem("b", "home"), mem("a", "home"), mem("c", "home"))
    trace = lexical().retrieve(state(*twins), query("home", limit=2))
    assert [c.memory_id for c in trace.candidates] == ["a", "b", "c"]
    assert len(trace.selected) == 2


def test_limit_caps_selection_not_candidates() -> None:
    trace = lexical().retrieve(state(*CORPUS), query("home", limit=1))
    assert len(trace.selected) == 1
    assert len(trace.candidates) == 3


@pytest.mark.parametrize("text", ["", "   ", "?!", "zebra"])
def test_nothing_eligible_means_abstain(text: str) -> None:
    s = state(*CORPUS)
    trace = lexical().retrieve(s, query(text))
    assert trace.selected == ()
    assert not any(c.eligible for c in trace.candidates)
    response = extractive(trace, s)
    assert (response.output, response.cited) == (None, ())


def test_empty_state() -> None:
    s = state()
    trace = lexical_recency().retrieve(s, query("home"))
    assert (trace.candidates, trace.selected) == ((), ())
    assert extractive(trace, s).output is None


def test_retrieval_is_deterministic() -> None:
    a = lexical_recency().retrieve(state(*CORPUS), query("home"))
    b = lexical_recency().retrieve(state(*reversed(CORPUS)), query("home"))
    assert a.digest == b.digest


def test_state_must_match_query_coordinates() -> None:
    with pytest.raises(ValueError, match="time coordinates"):
        lexical().retrieve(state(*CORPUS, at=10), query("home", at=11))


@dataclass(frozen=True)
class Broken:
    """A plugged-in signal that violates the contract."""

    drop: bool
    name: ClassVar[str] = "broken"

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return ()

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        out = [SignalEvidence(signal="broken", score=1.0) for _ in memories]
        return out[1:] if self.drop else [SignalEvidence(signal="other", score=1.0) for _ in out]


@pytest.mark.parametrize("drop", [True, False])
def test_contract_violating_signal_is_rejected(drop: bool) -> None:
    r = Retriever("r", ((Broken(drop), 1.0),), gate="broken")
    with pytest.raises(ValueError, match="exactly once"):
        r.retrieve(state(*CORPUS), query("home"))


def test_custom_signal_plugs_in_without_core_changes() -> None:
    @dataclass(frozen=True)
    class Length:
        name: ClassVar[str] = "length"

        @property
        def params(self) -> tuple[tuple[str, Scalar], ...]:
            return ()

        def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
            return [
                SignalEvidence(signal="length", score=float(len(m.content or ""))) for m in memories
            ]

    r = Retriever("bm25+length", ((BM25(), 1.0), (Length(), 0.01)), gate="bm25")
    trace = r.retrieve(state(*CORPUS), query("home"))
    assert [c.memory_id for c in trace.candidates][:2] == ["home", "tea"]
    assert RetrievalTrace.model_validate_json(trace.canonical()) == trace


# --- the pipeline over formed memory ----------------------------------------------------------


@pytest.fixture
def statement_log() -> Iterator[MemoryLog]:
    with MemoryLog(":memory:") as log:
        for step in relocation_year():
            form(log, step.experience, StatementPolicy(), recorded_at=step.recorded_at)
        yield log


def ask(log: MemoryLog, text: str, valid: float, known: float) -> str | None:
    q = Query(text=text, valid_at=day(valid), known_at=day(known), limit=3)
    return answer(log, lexical(), q)[1].output


def test_answers_follow_both_time_axes(statement_log: MemoryLog) -> None:
    assert ask(statement_log, "where is home", 57, 59) == "home = Paris"
    assert ask(statement_log, "where is home", 57, 60) == "home = Berlin"
    assert ask(statement_log, "where is home", 345, 345) == "home = Hamburg"
    assert ask(statement_log, "employer", 10, 60) == "employer = Acme"
    assert ask(statement_log, "employer", 10, 61) == "employer = Globex"
    assert ask(statement_log, "employer", 175, 175) is None  # forgotten, not yet relearned
    assert ask(statement_log, "employer", 345, 345) == "employer = Initech"
    assert ask(statement_log, "home", 0, -1) is None  # before anything was known


def test_episodic_baseline_exhibits_interference() -> None:
    """Measured behaviour, not a target: raw episodes let stale reports win."""
    with MemoryLog(":memory:") as log:
        for step in relocation_year():
            form(log, step.experience, EpisodicPolicy(), recorded_at=step.recorded_at)
        q = Query(text="where is home", valid_at=day(100), known_at=day(100), limit=3)
        trace, response = answer(log, lexical(), q)
    top = trace.candidates[:3]
    assert len({c.total for c in top}) == 1  # an exact three-way BM25 tie ...
    assert response.output == "set home = Munich"  # ... broken by memory_id, towards stale data


def test_answer_records_trace_and_response(statement_log: MemoryLog) -> None:
    q = Query(text="pet", valid_at=day(345), known_at=day(345), limit=1)
    trace, response = answer(statement_log, lexical_recency(), q)
    assert statement_log.record(RetrievalTrace, trace.digest) == trace
    assert response.responder == EXTRACTIVE
    assert response.cited == trace.selected
    assert answer(statement_log, lexical_recency(), q) == (trace, response)  # idempotent
    statement_log.verify()
