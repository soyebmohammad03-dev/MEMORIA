# MEMORIA

**Long-Horizon Memory Intelligence & Reliability Observatory**

MEMORIA is a research platform for experimentally studying how AI systems form, store,
retrieve, update, contradict, forget, contaminate, and reconstruct information over long
interaction horizons.

> What actually happens to an AI system's memory as experiences accumulate, change,
> conflict, and disappear over time?

MEMORIA is not a chatbot memory library. It treats a memory system as the *subject* of
controlled experiments. It feeds the subject experiences, applies interventions, and
records every transformation with provenance, so any result can be replayed and traced
back to its sources.

## Status

**Phase 0: foundation.** What exists today:

- The project constitution and architecture ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- Core provenance primitives: content hashing, immutable `Experience` and bitemporal,
  hash-linked `MemoryVersion` records with enforced invariants
- Tests, strict type checking, linting, and CI

No storage, retrieval, experiments, metrics, API, or UI exist yet. The roadmap is in
§8 of the architecture document. This README describes only what is implemented.

## Research areas (planned)

Memory formation · versioning & supersession · temporal validity · retrieval ·
correction · contradiction detection · forgetting & decay · contamination ·
interference · consistency · provenance & memory autopsy · replay · memory graphs ·
benchmarking · statistical evaluation · evidence-backed reporting.

## Principles

Evidence or silence · reproducible runs from hashed manifests · explicit uncertainty ·
controlled comparisons · immutable, content-addressed records · local-first, with
pluggable model providers and no required paid APIs.

## Development

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
