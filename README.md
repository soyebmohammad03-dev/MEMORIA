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

**Phase 3 complete: reproducible experiments.** What exists today:

- The project constitution and architecture ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- `memoria.core`: content-addressed, immutable records with enforced invariants:
  experiences, bitemporal hash-linked memory versions (create / update / correct /
  forget), formation decisions, retrieval traces, responses, datasets, probes with
  epistemic ground truth (`known` / `unknown` / `contested`), manifests and run records
- `memoria.store`: an append-only SQLite log that validates every append and reference,
  verifies digests on read, and exports to / loads from canonical JSON Lines
- `memoria.formation`: `episodic-v1` and `statement-v1` baseline policies, every decision
  recorded with a reason
- `memoria.retrieval`: BM25 and recency signals with per-candidate evidence, an
  extractive responder that abstains without evidence
- `memoria.interventions`: seeded `drop`, `delay`, `reorder`, `contaminate` and `inject`
  perturbations of an experience stream
- `memoria.artifacts`: a content-addressed, write-once local artifact store
- `memoria.experiments`: a registry and runner that turn a manifest into artifacts, and a
  `reproduce` check that re-runs a manifest and compares every artifact digest
- `memoria.scenarios`: a hand-built year-long dataset and a seeded generator
  (`drifting_facts`) with ground truth computed from the simulated world

Runs record answers next to expectations; scoring and statistics are Phase 4. No neural
embeddings, LLM providers, API or UI exist yet. The roadmap is in §8 of the architecture
document. This README describes only what is implemented.

```python
from memoria.artifacts import ArtifactStore
from memoria.core import InterventionSpec, RunManifest, RunOutcomes
from memoria.experiments import execute, reproduce
from memoria.retrieval import EXTRACTIVE, lexical_recency
from memoria.scenarios import drifting_facts

store = ArtifactStore("var/artifacts")
dataset = store.put_record(drifting_facts(seed=7))
manifest = RunManifest(
    name="contamination-1d",
    dataset=dataset,
    interventions=(
        InterventionSpec(
            name="contaminate",
            params=(("delay_days", 1), ("rate", 0.5), ("seed", 1), ("source", "contaminant")),
        ),
    ),
    policy="statement-v1",
    retriever=lexical_recency().spec,
    responder=EXTRACTIVE,
)
run = execute(manifest, store)  # every artifact is stored and named by digest
outcomes = store.get_record(RunOutcomes, run.outcomes)
for probe in outcomes.probes[:3]:
    print(probe.probe, probe.output, probe.expected.status, probe.expected.values)
print(reproduce(run.digest, store).reproduced)  # True: same manifest, same artifacts
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
