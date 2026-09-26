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

**Phase 2 complete: formation, retrieval and traceable responses.** What exists today:

- The project constitution and architecture ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- `memoria.core`: content-addressed, immutable records: experiences, bitemporal
  hash-linked memory versions with create / update / correct / forget semantics, formation
  decisions, queries, retrieval traces and responses, each with enforced invariants
- `memoria.store`: an append-only SQLite log that validates every append and every
  cross-record reference, rejects mutation at the database level, verifies digests on
  read, and replays deterministically
- `memoria.formation`: two deterministic baseline policies (`episodic-v1` stores raw
  episodes; `statement-v1` maintains keyed memories from an explicit statement language)
  whose every decision, including "change nothing", is recorded with a reason
- `memoria.retrieval`: BM25 and recency signals with per-candidate evidence, a
  fully-recorded ranking rule, and an extractive responder that abstains without evidence
- `memoria.scenarios`: a deterministic year-long experience stream covering updates, late
  reports, corrections, forgetting, relearning, conflicts, duplicates and noise

No neural embeddings, LLM providers, experiment runner, metrics, API or UI exist yet.
The roadmap is in §8 of the architecture document. This README describes only what is
implemented.

```python
from memoria.core import Query
from memoria.formation import StatementPolicy, form
from memoria.retrieval import answer, lexical
from memoria.scenarios import day, relocation_year
from memoria.store import MemoryLog

with MemoryLog(":memory:") as log:  # or a file path
    for step in relocation_year():
        form(log, step.experience, StatementPolicy(), recorded_at=step.recorded_at)

    # The user moved on day 55; the system learned it on day 60.
    for known in (59, 60):
        q = Query(text="where is home", valid_at=day(57), known_at=day(known), limit=1)
        trace, response = answer(log, lexical(), q)
        print(known, response.output, response.cited)  # 59: Paris, 60: Berlin

    top = trace.candidates[0]  # why it was chosen: every signal's score and raw inputs
    print(top.total, top.signals[0].inputs)
    print(log.version(response.cited[0]).derived_from)  # the experience(s) behind it
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
