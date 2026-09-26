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

**Phases 1–4 complete; Phase 5 in progress (slice 1).** What exists today:

- **Memory history** (`core`, `store`): content-addressed, immutable, bitemporal records;
  create / update / correct / forget; an append-only, verified SQLite log
- **Formation and retrieval** (`formation`, `retrieval`): `episodic-v1` and
  `statement-v1` policies with recorded decisions; BM25 and recency retrieval with
  per-candidate evidence; an extractive responder that abstains without evidence
- **Experiments** (`interventions`, `artifacts`, `experiments`, `scenarios`): seeded
  interventions, a write-once artifact store, a manifest runner with a `reproduce` check,
  and datasets with epistemic ground truth
- **Evaluation** (`comparison`, `taxonomy`, `statistics`, `evaluation`): exact answer
  readings, a provenance-based failure taxonomy with a retrieval-versus-memory locus,
  measurements that list their probes, Wilson intervals, and paired comparisons with
  Newcombe intervals, exact McNemar tests, Holm adjustment and underpowered flags
- **Semantic memory infrastructure** (`embeddings`, `semantic`): an embedding contract
  with recorded model identity, a deterministic reference embedder, and a verifiable
  semantic index artifact with exact search

The reference embedder hashes character n-grams: it measures spelling similarity, not
meaning, and exists so the infrastructure is testable without downloading a model. No
neural embedder, semantic ranking signal, consolidation, memory graph, API or UI exists
yet. This README describes only what is implemented.

```python
from memoria.artifacts import ArtifactStore
from memoria.embeddings import HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, form
from memoria.scenarios import day, relocation_year
from memoria.semantic import SemanticIndex
from memoria.store import MemoryLog

store = ArtifactStore("var/artifacts")
embedder = HashedNgramEmbedder(dimensions=256)  # identity: embedder.spec.digest

with MemoryLog(":memory:") as log:
    for step in relocation_year().steps:
        form(log, step.experience, EpisodicPolicy(), recorded_at=step.recorded_at)
    state = log.state_as_of(valid_at=day(345), known_at=day(345))

index = SemanticIndex.build(state, embedder, store)  # state, vectors, manifest stored
same = SemanticIndex.load(store, index.digest, embedder)  # other embedders are refused
same.verify(store, embedder)  # re-embeds the stored state; bytes must match
for hit in same.search("employer Initech", 3, embedder):
    print(hit.rank, hit.similarity, hit.memory_id, hit.version)  # version -> provenance
```

The Phase 4 evaluation workflow (run conditions, evaluate, compare with intervals) is
shown in `memoria.scenarios.conditions` and `memoria.evaluation.compare`.

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
