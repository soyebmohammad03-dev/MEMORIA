"""Manifest schema v2: a declared vector representation, without changing v1 identities."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactStore
from memoria.core import (
    EmbedderSpec,
    ExtensibleRecord,
    IndexSpec,
    RepresentationSpec,
    RetrievalTrace,
    RunManifest,
    RunRecord,
    content_hash,
)
from memoria.embeddings import HashedNgramEmbedder
from memoria.evaluation import EvaluationError, compare, evaluate
from memoria.experiments import DEFAULT_REGISTRY, UnknownComponentError, execute, reproduce
from memoria.neural import ModelUnavailableError, minilm_spec, neural_embedders
from memoria.retrieval import EXTRACTIVE, lexical, semantic
from memoria.scenarios import conditions, relocation_year
from memoria.store import MemoryLog
from memoria.vectors import hnsw_spec

RELOCATION = relocation_year()
HASHED = HashedNgramEmbedder()


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    s = ArtifactStore(tmp_path / "var")
    s.put_record(RELOCATION)
    return s


def v1() -> RunManifest:
    return RunManifest(
        name="lexical",
        dataset=RELOCATION.digest,
        policy="statement-v1",
        retriever=lexical().spec,
        responder=EXTRACTIVE,
    )


def v2(embedder: HashedNgramEmbedder = HASHED, index: IndexSpec | None = None) -> RunManifest:
    return RunManifest(
        name="semantic",
        dataset=RELOCATION.digest,
        policy="statement-v1",
        retriever=semantic(embedder).spec,
        responder=EXTRACTIVE,
        representation=RepresentationSpec(embedder=embedder.spec, index=index or IndexSpec()),
    )


# --- backward compatibility ------------------------------------------------------------------


def test_v1_manifests_keep_their_canonical_form_and_digest() -> None:
    m = v1()
    canonical = m.canonical()
    assert "representation" not in canonical
    as_v1 = {k: v for k, v in json.loads(canonical).items()}
    assert m.digest == content_hash(as_v1)  # the digest is over exactly the v1 fields
    assert RunManifest.model_validate_json(canonical) == m  # v1 JSON loads unchanged


# Computed by the committed pre-v2 code (a clean worktree of the Phase 5 slice-1 commit).
PRE_V2 = {
    "baseline_manifest": "sha256:2204c77d9be03e049bc63354921063c54a449aa5bea0a8011ddfac4d845e284d",
    "hashed_spec": "sha256:31db174862ed8fbfd7a4e542b3ea7b35bf3e64943847435957ef28fe09892fef",
}


def test_digests_match_the_pre_v2_code() -> None:
    baseline = conditions(RELOCATION.digest, "statement-v1")["baseline"]
    assert baseline.digest == PRE_V2["baseline_manifest"]
    assert HASHED.spec.digest == PRE_V2["hashed_spec"]


def test_evolved_fields_appear_only_when_set() -> None:
    m = v2()
    assert json.loads(m.canonical())["representation"]["embedder"]["name"] == "hashed-char-ngrams"
    assert m.digest != v1().digest
    spec = HASHED.spec
    for field in (
        "model",
        "pooling",
        "max_tokens",
        "truncation",
        "runtime",
        "batch_size",
        "tolerance",
    ):
        assert field not in json.loads(spec.canonical())  # reference specs unchanged by v2
    assert "tolerance" in json.loads(minilm_spec().canonical())


def test_extensible_record_mechanism() -> None:
    class Thing(ExtensibleRecord):
        _evolved = frozenset({"extra"})
        a: int
        extra: int | None = None

    assert Thing(a=1).digest == content_hash({"a": 1})
    assert Thing(a=1, extra=2).digest == content_hash({"a": 1, "extra": 2})
    assert Thing.model_validate_json('{"a": 1}') == Thing(a=1)


def test_index_spec_rules() -> None:
    IndexSpec()
    hnsw_spec()
    with pytest.raises(ValidationError, match="exact index is built in"):
        IndexSpec(kind="hnsw")
    with pytest.raises(ValidationError, match="exact index is built in"):
        IndexSpec(kind="exact", backend="faiss")
    with pytest.raises(ValidationError, match="takes no parameters"):
        IndexSpec(params=(("m", 16),))
    with pytest.raises(ValueError, match="positive"):
        hnsw_spec(m=0)


# --- declared representations in runs -----------------------------------------------------------


def test_semantic_run_executes_evaluates_and_reproduces(store: ArtifactStore) -> None:
    record = execute(v2(), store)
    assert store.get_record(RunManifest, record.manifest).representation is not None
    ev = evaluate(record.digest, store)
    assert len(ev.probes) == len(RELOCATION.probes)
    assert reproduce(record.digest, store).reproduced
    trace_evidence = ev.probes[0]
    assert trace_evidence.retrieval.candidates > 0


def test_semantic_evidence_records_cosine_and_embedder(store: ArtifactStore) -> None:
    record = execute(v2(), store)
    with MemoryLog.load(store.get(record.log)) as log:
        (trace, *_) = log.records(RetrievalTrace)
    signal = trace.candidates[0].signals[0]
    assert signal.signal == "semantic"
    assert "cosine" in dict(signal.inputs)
    assert dict(trace.retriever.signals[0].params)["embedder"] == HASHED.spec.digest


@pytest.mark.parametrize(
    ("manifest", "error", "message"),
    [
        (
            RunManifest.model_validate(v2().model_dump() | {"representation": None}),
            ValueError,
            "needs a declared representation",
        ),
        (
            RunManifest.model_validate(v1().model_dump() | {"representation": v2().representation}),
            ValueError,
            "no signal uses",
        ),
        (v2(index=hnsw_spec()), ValueError, "exact index"),
        (
            RunManifest.model_validate(
                v2().model_dump()
                | {
                    "representation": RepresentationSpec(
                        embedder=HashedNgramEmbedder(seed=9).spec
                    ).model_dump()
                }
            ),
            ValueError,
            "does not rebuild",
        ),
    ],
    ids=[
        "signal-without-representation",
        "representation-without-signal",
        "ann-in-runs",
        "mismatch",
    ],
)
def test_inconsistent_v2_manifests_fail_before_any_work(
    store: ArtifactStore, manifest: RunManifest, error: type[Exception], message: str
) -> None:
    before = sorted(store.root.rglob("*"))
    with pytest.raises(error, match=message):
        execute(manifest, store)
    assert sorted(store.root.rglob("*")) == before


def test_neural_run_never_falls_back(store: ArtifactStore, tmp_path: Path) -> None:
    neural = RunManifest.model_validate(
        v2().model_dump()
        | {
            "retriever": v2()
            .retriever.model_copy(
                update={
                    "signals": (
                        v2()
                        .retriever.signals[0]
                        .model_copy(update={"params": (("embedder", minilm_spec().digest),)}),
                    )
                }
            )
            .model_dump(),
            "representation": RepresentationSpec(embedder=minilm_spec()).model_dump(),
        }
    )
    before = sorted(store.root.rglob("*"))
    with pytest.raises(UnknownComponentError, match="onnx-sentence"):
        execute(neural, store)  # not registered: no substitute is used
    with pytest.raises(ModelUnavailableError):
        execute(neural, store, DEFAULT_REGISTRY.extend(embedders=neural_embedders(tmp_path)))
    assert sorted(store.root.rglob("*")) == before
    assert store.digests("RunRecord") == []  # nothing claims a neural run happened


# --- comparisons account for the representation ------------------------------------------------


def test_representation_is_part_of_the_retrieval_variable(store: ArtifactStore) -> None:
    lexical_run = evaluate(execute(v1(), store).digest, store)
    semantic_run = evaluate(execute(v2(), store).digest, store)
    assert compare(lexical_run.digest, semantic_run.digest, store).variable == "retrieval"
    other_embedder = HashedNgramEmbedder(dimensions=128)
    only_embedding = evaluate(execute(v2(other_embedder), store).digest, store)
    c = compare(semantic_run.digest, only_embedding.digest, store)
    assert c.variable == "retrieval"
    drop = conditions(RELOCATION.digest, "statement-v1")["drop"]
    both = RunManifest.model_validate(
        drop.model_dump()
        | {"retriever": v2().retriever.model_dump(), "representation": v2().representation}
    )
    with pytest.raises(EvaluationError, match="exactly one"):
        compare(lexical_run.digest, evaluate(execute(both, store).digest, store).digest, store)


def test_embedder_resolution_is_round_tripped() -> None:
    assert DEFAULT_REGISTRY.embedder(HASHED.spec).spec == HASHED.spec
    tampered = EmbedderSpec.model_validate(
        HASHED.spec.model_dump() | {"params": (("n_max", 5), ("n_min", 3), ("seed", 0), ("x", 1))}
    )
    with pytest.raises(ValueError, match="does not describe"):
        DEFAULT_REGISTRY.embedder(tampered)


def test_run_records_are_unchanged_in_shape() -> None:
    assert set(RunRecord.model_fields) == {
        "manifest",
        "dataset",
        "interventions",
        "log",
        "outcomes",
    }
