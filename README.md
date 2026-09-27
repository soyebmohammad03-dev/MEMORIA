# MEMORIA

**Long-Horizon Memory Intelligence & Reliability Observatory**

MEMORIA is an experimental laboratory and observatory for studying the formation,
representation, consolidation, retrieval, revision, forgetting, interference,
contamination, provenance and reliability of artificial memory over long interaction
horizons.

> What actually happens to an AI system's memory as experiences accumulate, change,
> conflict, and disappear over time?

MEMORIA is not a chatbot memory library. It treats a memory system as the *subject* of
controlled experiments: it feeds the subject experiences, applies interventions, and
records every transformation with provenance, so any result can be replayed, measured
with its uncertainty, and traced back to its sources. The goal is to compare memory
architectures and policies scientifically — for example, what becomes durable memory,
when similar memories interfere, how contradictions revise beliefs, how resilient memory
is to contamination, and exactly why a memory influenced an answer. The full research
questions (RQ1–RQ15), target architecture and staged roadmap to Phase 22 are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Status

**Phases 1–7 complete.** What exists today:

- **Memory history** (`core`, `store`): content-addressed, immutable, bitemporal records;
  create / update / correct / forget; an append-only, verified SQLite log
- **Formation and retrieval** (`formation`, `retrieval`): `episodic-v1` and
  `statement-v1` policies with recorded decisions; lexical (BM25), recency and semantic
  signals with per-candidate evidence; an extractive responder that abstains without
  evidence
- **Experiments** (`interventions`, `artifacts`, `experiments`, `scenarios`): seeded
  interventions, a write-once artifact store, a manifest runner with a `reproduce` check,
  datasets with epistemic ground truth; manifest schema v2 declares the vector
  representation (embedder and index) as an experimental variable
- **Evaluation** (`comparison`, `taxonomy`, `statistics`, `evaluation`): a
  provenance-based failure taxonomy, measurements that list their probes, Wilson
  intervals, paired Newcombe/McNemar comparisons with Holm adjustment
- **Semantic memory** (`embeddings`, `neural`, `semantic`, `vectors`, `semantic_eval`):
  a deterministic lexical-subword reference embedder; a local neural sentence embedder
  (all-MiniLM-L6-v2 on ONNX Runtime, pinned by revision and file hashes) as an optional
  extra; verifiable semantic indexes with exact search and optional FAISS HNSW candidate
  generation; a controlled lexical-vs-neural study
  ([results](docs/experiments/phase5-semantic.md))
- **Hybrid retrieval** (`hybrid`, `hybrid_eval`): candidate generation (lexical, semantic,
  metadata), recorded hard filters, nine signals with explicit missing values, declared
  normalisation, weighted and rank-fusion policies, MMR diversity and contradiction
  exposure, all as content-addressed policies whose traces re-derive every score and
  explanation; ablations, leave-one-out, counterfactuals and adversarial cases with paired
  statistics ([results](docs/experiments/phase6-hybrid.md))
- **Consolidation** (`consolidation`, `consolidation_eval`): derived, hierarchical memory
  (L2 facts, L3 entity profiles, L4 timelines and inferred patterns) built as
  content-addressed artifacts over immutable history, never as evidence; guarded
  deduplication that reports unsafe merges; temporal consolidation; decomposed importance;
  structured information-loss reports; consolidation and hybrid retrieval as run-manifest
  variables (schema v3); a stability–plasticity laboratory
  ([results](docs/experiments/phase7-consolidation.md))

Neural embeddings are an experimental representation, not ground truth: in the Phase 5
study they rate numeric changes and contradictions as similar to a query as true
paraphrases. In the Phase 6 study, a superseded claim outranked the current one under
policies that surface rather than penalise conflict. Retrieval relevance is a policy
score, not truth. Consolidation's gains and losses are both measured: in the Phase 7 lab,
temporal consolidation raised correct answers while exact deduplication merged identical
statements from different periods. No memory graph, API or UI exists yet. This README
describes only what is implemented.

The core needs no neural dependencies. For the neural representation and approximate
search, install the extras and fetch the pinned model explicitly (nothing downloads
implicitly):

```bash
uv sync --extra neural --extra ann
```

```python
from memoria.artifacts import ArtifactStore
from memoria.embeddings import HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, form
from memoria.neural import (
    MINILM,
    NeuralSentenceEmbedder,
    default_model_dir,
    fetch_model,
    minilm_spec,
)
from memoria.scenarios import semantic_diagnostic
from memoria.semantic import SemanticIndex
from memoria.store import MemoryLog

fetch_model(MINILM, default_model_dir())  # once: pinned files, verified by SHA-256
neural = NeuralSentenceEmbedder.open(minilm_spec(), default_model_dir())
dataset, _ = semantic_diagnostic()
with MemoryLog(":memory:") as log:
    for step in dataset.steps:
        form(log, step.experience, EpisodicPolicy(), recorded_at=step.recorded_at)
    end = dataset.steps[-1].recorded_at
    state = log.state_as_of(valid_at=end, known_at=end)

store = ArtifactStore("var/artifacts")
for embedder in (HashedNgramEmbedder(), neural):  # control condition, then neural
    index = SemanticIndex.build(state, embedder, store)  # vectors, state, manifest stored
    for hit in index.search("What city is Ana's home?", 3, embedder):
        print(embedder.spec.name, hit.rank, round(hit.similarity, 3), hit.version[:16])
```

The Phase 5 study is one command: `python -m memoria.semantic_eval var/artifacts`; the
Phase 6 study is `python -m memoria.hybrid_eval var/artifacts minilm` (or `hashed`, which
needs no model). The Phase 7 lab is `python -m memoria.consolidation_eval var/artifacts`.

## Research areas

Memory formation · versioning & supersession · temporal validity · retrieval ·
correction · contradiction detection · forgetting & decay · contamination ·
interference · consistency · provenance & memory autopsy · replay · memory graphs ·
benchmarking · statistical evaluation · evidence-backed reporting.

## Principles

Evidence or silence · reproducible runs from hashed manifests · explicit uncertainty ·
controlled comparisons · immutable, content-addressed records · local-first, with
pluggable model providers and no required paid APIs.

## License

[MIT](LICENSE)

## Development

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
