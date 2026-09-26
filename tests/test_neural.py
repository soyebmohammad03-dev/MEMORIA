"""The neural embedder adapter: contract, identity, file integrity, availability.

Most tests use :class:`ArithmeticTestBackend`, a deterministic test double that derives
token vectors from token ids. It is not a model and is never labelled as one: its spec
names a test model. Tests marked ``model`` run the pinned MiniLM files when present.
"""

import hashlib
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.core import EmbedderSpec, ModelFile, ModelIdentity, Tolerance
from memoria.embeddings import Agreement, agreement, cosine, embed
from memoria.experiments import DEFAULT_REGISTRY, UnknownComponentError
from memoria.neural import (
    MINILM,
    MINILM_CARD,
    NAME,
    RUNTIME,
    VERSION,
    ModelIntegrityError,
    ModelUnavailableError,
    NeuralSentenceEmbedder,
    Tokens,
    default_model_dir,
    minilm_spec,
    neural_embedders,
    verify_files,
)

TEST_MODEL = ModelIdentity(
    provider="test",
    id="memoria/arithmetic-test-double",
    revision="1",
    files=(ModelFile(path="model.bin", sha256="0" * 64, size=0),),
)


def spec(**changes: object) -> EmbedderSpec:
    fields: dict[str, object] = {
        "name": NAME,
        "version": VERSION,
        "dimensions": 4,
        "normalized": True,
        "model": TEST_MODEL,
        "pooling": "mean",
        "max_tokens": 8,
        "truncation": "right",
        "runtime": RUNTIME,
        "batch_size": 1,
        "tolerance": Tolerance(max_abs=1e-6, min_cosine=0.99999),
    }
    return EmbedderSpec.model_validate(fields | changes)


class ArithmeticTestBackend:
    """Test double: token i of text t gets vector (id, i, len, 1) — no semantics at all."""

    def __init__(self, broken: bool = False) -> None:
        self.broken = broken
        self.seen_max_tokens: list[int] = []

    def tokenize(self, text: str, max_tokens: int) -> Tokens:
        self.seen_max_tokens.append(max_tokens)
        ids = [101, *(ord(c) for c in text), 102][:max_tokens]
        mask = [1] * len(ids)
        if text.endswith("~"):  # simulate padding the model must ignore
            ids, mask = [*ids, 0, 0], [*mask, 0, 0]
        return Tokens(ids, mask, [0] * len(ids))

    def token_embeddings(self, tokens: Tokens) -> Sequence[Sequence[float]]:
        rows = [[float(t), float(i), float(len(tokens.ids)), 1.0] for i, t in enumerate(tokens.ids)]
        return rows[:-1] if self.broken else rows


# --- adapter contract ---------------------------------------------------------------------------


def test_mean_pooling_normalisation_and_truncation() -> None:
    backend = ArithmeticTestBackend()
    e = NeuralSentenceEmbedder(spec(), backend)
    (v,) = embed(e, ["ab"])
    tokens = [101, 97, 98, 102]
    pooled = [sum(tokens) / 4, 1.5, 4.0, 1.0]
    norm = math.sqrt(sum(x * x for x in pooled))
    assert v == pytest.approx([x / norm for x in pooled])
    assert backend.seen_max_tokens == [8]  # the spec's max_tokens reaches the tokenizer


def test_padding_is_excluded_from_pooling() -> None:
    e = NeuralSentenceEmbedder(spec(normalized=False), ArithmeticTestBackend())
    (plain,) = e.encode(["ab"])
    (padded,) = e.encode(["ab~"])
    assert padded[0] == pytest.approx((101 + 97 + 98 + 126 + 102) / 5)  # masked zeros ignored
    assert plain != padded


def test_cls_pooling_uses_the_first_token() -> None:
    e = NeuralSentenceEmbedder(spec(pooling="cls", normalized=False), ArithmeticTestBackend())
    assert e.encode(["xyz"]) == [[101.0, 0.0, 5.0, 1.0]]


def test_token_count_mismatch_is_an_integrity_error() -> None:
    e = NeuralSentenceEmbedder(spec(), ArithmeticTestBackend(broken=True))
    with pytest.raises(ModelIntegrityError, match="token vectors"):
        e.encode(["ab"])


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other"},
        {"version": "2"},
        {"runtime": "torch"},
        {"batch_size": 8},
    ],
)
def test_adapter_refuses_specs_it_does_not_implement(changes: dict[str, object]) -> None:
    with pytest.raises(ModelIntegrityError):
        NeuralSentenceEmbedder(spec(**changes), ArithmeticTestBackend())


# --- identity -------------------------------------------------------------------------------------


def test_model_specs_must_be_complete() -> None:
    for missing in ("pooling", "max_tokens", "truncation", "runtime", "batch_size", "tolerance"):
        with pytest.raises(ValidationError, match="must declare"):
            spec(**{missing: None})
    with pytest.raises(ValidationError, match="require a model identity"):
        spec(model=None)
    with pytest.raises(ValidationError, match="unique and sorted"):
        ModelIdentity.model_validate(
            TEST_MODEL.model_dump() | {"files": [TEST_MODEL.files[0].model_dump()] * 2}
        )
    with pytest.raises(ValidationError):
        ModelFile(path="x", sha256="not-hex", size=1)
    with pytest.raises(ValidationError):
        Tolerance(max_abs=0, min_cosine=0.9)


def test_identity_changes_with_every_output_affecting_choice() -> None:
    base = minilm_spec()
    revised = MINILM.model_copy(update={"revision": "0" * 40})
    reweighted = MINILM.model_copy(
        update={"files": (MINILM.files[0].model_copy(update={"sha256": "1" * 64}), MINILM.files[1])}
    )
    variants = [
        base.model_copy(update={"model": revised}),
        base.model_copy(update={"model": reweighted}),
        base.model_copy(update={"pooling": "cls"}),
        base.model_copy(update={"max_tokens": 128}),
        base.model_copy(update={"normalized": False}),
        base.model_copy(update={"tolerance": Tolerance(max_abs=1e-3, min_cosine=0.999)}),
    ]
    assert len({base.digest, *(v.digest for v in variants)}) == len(variants) + 1


def test_minilm_identity_is_pinned() -> None:
    s = minilm_spec()
    assert s.model is not None
    assert (s.model.id, s.model.revision) == (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )
    assert (s.dimensions, s.pooling, s.max_tokens, s.normalized) == (384, "mean", 256, True)
    assert not s.preprocessing.casefold  # the model's own tokenizer lowercases
    assert s.digest == "sha256:ec4571c402e2159015075aae3051c2394e77b4dc58c67a7edce6b6961866236c"
    assert EmbedderSpec.model_validate_json(s.canonical()) == s


def test_license_is_metadata_not_identity() -> None:
    assert MINILM_CARD.license == "apache-2.0"
    assert MINILM_CARD.model == MINILM.digest
    assert "license" not in minilm_spec().canonical()


# --- files, availability, no fallback -------------------------------------------------------------


def fake_model(tmp_path: Path, data: bytes = b"weights") -> ModelIdentity:
    (tmp_path / "model.bin").write_bytes(data)
    return ModelIdentity(
        provider="test",
        id="memoria/fake",
        revision="1",
        files=(ModelFile(path="model.bin", sha256=hashlib.sha256(b"weights").hexdigest(), size=7),),
    )


def test_file_verification(tmp_path: Path) -> None:
    identity = fake_model(tmp_path)
    verify_files(tmp_path, identity)
    (tmp_path / "model.bin").write_bytes(b"weightz")
    with pytest.raises(ModelIntegrityError, match="sha256"):
        verify_files(tmp_path, identity)
    (tmp_path / "model.bin").write_bytes(b"weights!")
    with pytest.raises(ModelIntegrityError, match="size"):
        verify_files(tmp_path, identity)
    (tmp_path / "model.bin").unlink()
    with pytest.raises(ModelUnavailableError, match=r"missing model\.bin"):
        verify_files(tmp_path, identity)


def test_missing_model_fails_diagnostically(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="fetch it explicitly"):
        NeuralSentenceEmbedder.open(minilm_spec(), tmp_path / "nowhere")


def test_no_silent_fallback_through_the_registry(tmp_path: Path) -> None:
    with pytest.raises(UnknownComponentError, match="embedder 'onnx-sentence'"):
        DEFAULT_REGISTRY.embedder(minilm_spec())
    registry = DEFAULT_REGISTRY.extend(embedders=neural_embedders(tmp_path))
    with pytest.raises(ModelUnavailableError):
        registry.embedder(minilm_spec())


def test_core_works_without_the_neural_runtime() -> None:
    """With onnxruntime and tokenizers blocked, the core and reference path still work,
    and the neural adapter fails with a diagnostic instead of degrading."""
    code = """
import sys
sys.modules["onnxruntime"] = None
sys.modules["tokenizers"] = None
from pathlib import Path
from memoria.embeddings import HashedNgramEmbedder, embed
from memoria.neural import OnnxBackend, ModelUnavailableError
assert len(embed(HashedNgramEmbedder(), ["still works"])[0]) == 256
try:
    OnnxBackend(Path("."), "model.onnx", "tokenizer.json")
except ModelUnavailableError as e:
    print("unavailable:", "install the 'neural' extra" in str(e))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "unavailable: True"


def test_default_model_dir_is_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMORIA_MODELS", str(tmp_path))
    assert default_model_dir() == tmp_path / "all-MiniLM-L6-v2" / MINILM.revision
    assert default_model_dir(root=tmp_path / "x").parent.parent == tmp_path / "x"


# --- tolerance ----------------------------------------------------------------------------------


def test_agreement_classes() -> None:
    t = Tolerance(max_abs=1e-4, min_cosine=0.9999)
    a = [[0.6, 0.8], [1.0, 0.0]]
    assert agreement(a, [[0.6, 0.8], [1.0, 0.0]], t) is Agreement.IDENTICAL
    assert agreement(a, [[0.60005, 0.79996], [1.0, 0.0]], t) is Agreement.EQUIVALENT
    assert agreement(a, [[0.60005, 0.79996], [1.0, 0.0]], None) is Agreement.DIFFERENT
    assert agreement(a, [[0.61, 0.79], [1.0, 0.0]], t) is Agreement.DIFFERENT
    assert agreement(a, [[0.6, 0.8]], t) is Agreement.DIFFERENT
    assert agreement([[0.0, 0.0]], [[0.0, 0.00001]], t) is Agreement.DIFFERENT


# --- the real model (skipped without the pinned files and the neural extra) ----------------------


def _real() -> NeuralSentenceEmbedder:
    try:
        return NeuralSentenceEmbedder.open(minilm_spec(), default_model_dir())
    except ModelUnavailableError as e:
        pytest.skip(f"pinned model unavailable: {e}")


@pytest.mark.model
def test_real_model_shape_determinism_and_semantics() -> None:
    e = _real()
    texts = ["Ana lives in Berlin.", "Ana's home is in Berlin.", "The cat sleeps on the mat."]
    first = embed(e, texts)
    assert all(len(v) == 384 for v in first)
    assert all(math.isclose(math.fsum(x * x for x in v), 1.0, rel_tol=1e-5) for v in first)
    assert agreement(first, embed(e, texts), e.spec.tolerance) is Agreement.IDENTICAL
    assert cosine(first[0], first[1]) > cosine(first[0], first[2]) + 0.3


@pytest.mark.model
def test_real_model_truncates_at_max_tokens_and_handles_unicode() -> None:
    e = _real()
    long, longer = embed(e, ["word " * 300, "word " * 400])
    assert long == longer  # both exceed 256 tokens: identical after right truncation
    zoe, zoe_nfd = embed(e, ["Zoë's café", "Zoë's café"])
    assert agreement([zoe], [zoe_nfd], e.spec.tolerance) is not Agreement.DIFFERENT  # NFKC first


@pytest.mark.model
def test_runtime_telemetry_is_disabled_before_onnxruntime_initialises() -> None:
    """ONNX Runtime's POSIX build ships an HTTP telemetry client; MEMORIA opts out before
    the runtime initialises (the opt-out is read at initialisation)."""
    code = """
import os, sys
assert "onnxruntime" not in sys.modules
from memoria.neural import NeuralSentenceEmbedder, default_model_dir, minilm_spec
try:
    NeuralSentenceEmbedder.open(minilm_spec(), default_model_dir())
except Exception as e:
    print("skip", type(e).__name__)
else:
    print(os.environ.get("ORT_DISABLE_TELEMETRY"))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    if out.stdout.startswith("skip"):
        pytest.skip(f"pinned model unavailable ({out.stdout.strip()})")
    assert out.stdout.strip() == "1"
