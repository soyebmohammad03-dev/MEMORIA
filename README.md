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

**Phase 1 complete: memory history.** What exists today:

- The project constitution and architecture ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- `memoria.core`: content-addressed, immutable `Experience` and bitemporal, hash-linked
  `MemoryVersion` records; explicit create / update / correct / forget semantics; a pure
  fold answering "what did the system believe at record time *k* about valid time *t*?"
- `memoria.store`: an append-only SQLite operation log that validates every append,
  rejects mutation at the database level, verifies digests on read, and replays
  deterministically
- Tests, strict type checking, linting, and CI

No formation, retrieval, experiments, metrics, API, or UI exist yet. The roadmap is in
§8 of the architecture document. This README describes only what is implemented.

```python
from datetime import UTC, datetime
from memoria.core import Experience, MemoryVersion, Operation
from memoria.store import MemoryLog

t = lambda d: datetime(2026, 1, d, tzinfo=UTC)
with MemoryLog(":memory:") as log:  # or a file path
    said = log.append_experience(Experience(source="chat:1", content="I live in Paris.", occurred_at=t(1)))
    v1 = MemoryVersion(memory_id="home", version=1, operation=Operation.CREATE, content="Paris",
                       valid_from=t(1), recorded_at=t(1), derived_from=(said,))
    log.append_version(v1)
    log.append_version(v1.successor(Operation.UPDATE, content="Berlin", valid_from=t(20), recorded_at=t(28)))
    log.state_as_of(valid_at=t(25), known_at=t(27))  # Paris: the move was not yet known
    log.state_as_of(valid_at=t(25), known_at=t(28))  # Berlin: learned on day 28, true from day 20
```

## Research areas (planned)

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
