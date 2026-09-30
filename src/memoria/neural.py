"""Local neural sentence embeddings (optional extra: ``memoria[neural]``).

:class:`NeuralSentenceEmbedder` runs a transformer encoder locally and pools its token
embeddings into one vector per text, exactly as its :class:`~memoria.core.EmbedderSpec`
declares: tokenise (truncating on the right at ``max_tokens``), encode one text at a
time, mean-pool over the attention mask, L2-normalise. Before any inference it verifies
every pinned model file by size and SHA-256; nothing is downloaded implicitly.

:class:`OnnxBackend` is the runtime: ONNX Runtime on the CPU (one thread, sequential
execution) with a Hugging Face ``tokenizers`` tokenizer. Its imports are deferred, so
the core package never needs them. A neural embedding is an experimental
representation, not ground truth: it is only as good as the model, and it is
reproducible only within the tolerance its spec declares (see ARCHITECTURE.md §4.8).

:data:`MINILM` pins ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0, 384
dimensions, mean pooling, 256 tokens) to an immutable repository revision and the hashes
of the files that determine its output; :func:`fetch_model` downloads exactly those.
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

from memoria.core import (
    EmbedderSpec,
    ModelFile,
    ModelIdentity,
    Preprocessing,
    Record,
    Tolerance,
)

NAME = "onnx-sentence"
VERSION = "1"
RUNTIME = "onnxruntime-cpu"
EmbedderFactory = Callable[[EmbedderSpec], "NeuralSentenceEmbedder"]


class ModelUnavailableError(RuntimeError):
    """The model or its runtime is not available locally. Nothing falls back silently."""


class ModelIntegrityError(RuntimeError):
    """Local model files are not the bytes the spec pins, or model metadata is malformed."""


class Tokens(NamedTuple):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class TransformerBackend(Protocol):
    """Tokeniser plus encoder. Must be deterministic for a given input."""

    def tokenize(self, text: str, max_tokens: int) -> Tokens: ...

    def token_embeddings(self, tokens: Tokens) -> Sequence[Sequence[float]]:
        """One vector per token, in order."""
        ...


class ModelCard(Record):
    """Descriptive metadata about a model. Not part of its identity: none of it changes
    what the model computes."""

    model: str  # ModelIdentity digest
    license: str
    source: str  # where the pinned revision can be obtained
    description: str


def _check_file(path: Path, f: ModelFile) -> None:
    if path.stat().st_size != f.size:
        raise ModelIntegrityError(f"{f.path}: size {path.stat().st_size}, pinned {f.size}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != f.sha256:
        raise ModelIntegrityError(f"{f.path}: sha256 {h.hexdigest()}, pinned {f.sha256}")


def verify_files(model_dir: Path, identity: ModelIdentity) -> None:
    """Every pinned file must exist with the pinned size and SHA-256."""
    for f in identity.files:
        path = model_dir / f.path
        if not path.is_file():
            raise ModelUnavailableError(
                f"{identity.id}@{identity.revision}: missing {f.path} in {model_dir} "
                "(fetch it explicitly with memoria.neural.fetch_model)"
            )
        _check_file(path, f)


class OnnxBackend:
    """ONNX Runtime encoder and ``tokenizers`` tokenizer, configured for determinism."""

    def __init__(self, model_dir: Path, model_file: str, tokenizer_file: str) -> None:
        # ONNX Runtime's POSIX build ships a telemetry client (Microsoft 1DS) that uploads
        # over HTTP; MEMORIA is local-first and allows no hidden network traffic. The
        # opt-out is read when onnxruntime initialises, so it is set before the import.
        # (If the host imported onnxruntime earlier without it, only the per-event switch
        # below applies — see ARCHITECTURE.md §4.6.)
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ModelUnavailableError(
                "the neural runtime is not installed; install the 'neural' extra "
                "(onnxruntime, tokenizers, numpy)"
            ) from e
        ort.disable_telemetry_events()
        self._tokenizer = Tokenizer.from_file(str(model_dir / tokenizer_file))
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_dir / model_file), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def tokenize(self, text: str, max_tokens: int) -> Tokens:
        self._tokenizer.enable_truncation(max_length=max_tokens, direction="right")
        enc = self._tokenizer.encode(text)
        return Tokens(list(enc.ids), list(enc.attention_mask), list(enc.type_ids))

    def token_embeddings(self, tokens: Tokens) -> Sequence[Sequence[float]]:
        import numpy as np

        feed = {
            "input_ids": np.array([tokens.ids], dtype=np.int64),
            "attention_mask": np.array([tokens.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([tokens.type_ids], dtype=np.int64),
        }
        (hidden,) = self._session.run(
            ["last_hidden_state"], {k: v for k, v in feed.items() if k in self._inputs}
        )
        rows: list[list[float]] = hidden[0].tolist()
        return rows


class NeuralSentenceEmbedder:
    """A transformer sentence embedder whose every output-affecting choice is in its spec."""

    def __init__(self, spec: EmbedderSpec, backend: TransformerBackend) -> None:
        if spec.name != NAME or spec.version != VERSION:
            raise ModelIntegrityError(f"spec names adapter {spec.name} v{spec.version}")
        if spec.model is None or spec.max_tokens is None or spec.pooling is None:
            raise ModelIntegrityError("a neural spec must identify its model")
        if (spec.runtime, spec.batch_size, spec.truncation) != (RUNTIME, 1, "right"):
            raise ModelIntegrityError(
                f"this adapter runs {RUNTIME}, one text per batch, truncating right"
            )
        self._spec = spec
        self._backend = backend

    @classmethod
    def open(cls, spec: EmbedderSpec, model_dir: Path) -> NeuralSentenceEmbedder:
        """Verify the pinned files in ``model_dir``, then load the ONNX runtime."""
        if spec.model is None:
            raise ModelIntegrityError("a neural spec must identify its model")
        verify_files(model_dir, spec.model)
        paths = {f.path for f in spec.model.files}
        model_file = next((p for p in sorted(paths) if p.endswith(".onnx")), None)
        if model_file is None or "tokenizer.json" not in paths:
            raise ModelIntegrityError("a neural spec must pin one .onnx file and tokenizer.json")
        return cls(spec, OnnxBackend(model_dir, model_file, "tokenizer.json"))

    @property
    def spec(self) -> EmbedderSpec:
        return self._spec

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        assert self._spec.max_tokens is not None  # checked in __init__
        tokens = self._backend.tokenize(text, self._spec.max_tokens)
        rows = self._backend.token_embeddings(tokens)
        if len(rows) != len(tokens.ids):
            raise ModelIntegrityError(f"{len(rows)} token vectors for {len(tokens.ids)} tokens")
        if self._spec.pooling == "cls":
            pooled = list(rows[0])
        else:
            kept = [r for r, m in zip(rows, tokens.attention_mask, strict=True) if m]
            if not kept:
                raise ModelIntegrityError("no attended tokens to pool")
            pooled = [math.fsum(col) / len(kept) for col in zip(*kept, strict=True)]
        if not self._spec.normalized:
            return pooled
        norm = math.sqrt(math.fsum(x * x for x in pooled))
        return [x / norm for x in pooled] if norm else pooled


# --- the pinned reference model -------------------------------------------------------------

MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM = ModelIdentity(
    provider="huggingface",
    id=MINILM_ID,
    revision=MINILM_REVISION,
    files=(
        ModelFile(
            path="onnx/model.onnx",
            sha256="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
            size=90405214,
        ),
        ModelFile(
            path="tokenizer.json",
            sha256="be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
            size=466247,
        ),
    ),
)
MINILM_CARD = ModelCard(
    model=MINILM.digest,
    license="apache-2.0",
    source=f"https://huggingface.co/{MINILM_ID}/tree/{MINILM_REVISION}",
    description="MiniLM-L6 sentence encoder, 384 dimensions; published pipeline: "
    "transformer, mean pooling over tokens, L2 normalisation; max_seq_length 256.",
)
# Float32 inference differs across CPUs and runtime versions in the last few bits
# (kernel choice, accumulation order). Outputs within these bounds are equivalent; the
# bounds are ~100x the differences observed between repeated runs and far below the
# similarity gaps that separate retrieval ranks (both measured, ARCHITECTURE.md §4.8).
MINILM_TOLERANCE = Tolerance(max_abs=1e-4, min_cosine=0.9999)


def minilm_spec() -> EmbedderSpec:
    """The published all-MiniLM-L6-v2 pipeline. The tokenizer lowercases and normalises
    Unicode itself, so MEMORIA's preprocessing only applies NFKC and collapses whitespace."""
    return EmbedderSpec(
        name=NAME,
        version=VERSION,
        dimensions=384,
        preprocessing=Preprocessing(unicode="NFKC", casefold=False, collapse_whitespace=True),
        normalized=True,
        model=MINILM,
        pooling="mean",
        max_tokens=256,
        truncation="right",
        runtime=RUNTIME,
        batch_size=1,
        tolerance=MINILM_TOLERANCE,
    )


def neural_embedders(model_root: Path | None = None) -> dict[str, EmbedderFactory]:
    """Registry entries for neural embedders (``Registry.extend(embedders=...)``).

    ``model_root`` is execution environment, not identity: files are found at
    ``<root>/<model name>/<revision>`` and verified against the spec before use.
    """

    def open_spec(spec: EmbedderSpec) -> NeuralSentenceEmbedder:
        if spec.model is None:
            raise ModelIntegrityError("a neural spec must identify its model")
        return NeuralSentenceEmbedder.open(spec, default_model_dir(spec.model, model_root))

    return {NAME: open_spec}


def default_model_dir(identity: ModelIdentity = MINILM, root: Path | None = None) -> Path:
    """``$MEMORIA_MODELS/<name>/<revision>``, default ``~/.cache/memoria/models``."""
    root = root or Path(
        os.environ.get("MEMORIA_MODELS", Path.home() / ".cache" / "memoria" / "models")
    )
    return root / identity.id.rsplit("/", 1)[-1] / identity.revision


def fetch_model(identity: ModelIdentity, model_dir: Path) -> Path:
    """Download exactly the pinned files at the pinned revision and verify them.

    Explicit only: nothing in MEMORIA calls this during a run or an evaluation.
    """
    if identity.provider != "huggingface":
        raise ModelUnavailableError(f"cannot fetch from provider {identity.provider!r}")
    for f in identity.files:
        target = model_dir / f.path
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{identity.id}/resolve/{identity.revision}/{f.path}"
        partial = target.with_name(target.name + ".partial")
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        try:
            _check_file(partial, f)  # a corrupt download must never become the pinned file
        except ModelIntegrityError:
            partial.unlink()
            raise
        partial.replace(target)
    verify_files(model_dir, identity)
    return model_dir
