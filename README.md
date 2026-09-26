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

**Phase 4 complete: evaluation and failure analysis.** What exists today:

- The project constitution and architecture ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- **Memory history** (`core`, `store`): content-addressed, immutable, bitemporal records;
  create / update / correct / forget; an append-only, verified SQLite log
- **Formation and retrieval** (`formation`, `retrieval`): `episodic-v1` and
  `statement-v1` policies with recorded decisions; BM25 and recency retrieval with full
  per-candidate evidence; an extractive responder that abstains without evidence
- **Experiments** (`interventions`, `artifacts`, `experiments`, `scenarios`): seeded
  `drop` / `delay` / `reorder` / `contaminate` / `inject` interventions, a write-once
  artifact store, a manifest runner with a `reproduce` check, and datasets with
  epistemic ground truth (`known` / `unknown` / `contested`)
- **Evaluation** (`comparison`, `taxonomy`, `statistics`, `evaluation`): exact answer
  readings (statement, assignment, natural-language mention); a provenance-based failure
  taxonomy (stale, contaminated, correction failure, temporal error, contradiction,
  wrong memory, forgotten recalled, missing memory, retrieval miss, unsupported, ...) with
  the rule that fired and a retrieval-versus-memory locus; measurements that list their
  probes; Wilson intervals; paired baseline-vs-treatment comparisons with Newcombe
  intervals, exact McNemar tests, Holm adjustment and underpowered flags

No neural embeddings, LLM providers, provenance graph, API or UI exist yet. The roadmap
is in §8 of the architecture document. This README describes only what is implemented.

```python
from memoria.artifacts import ArtifactStore
from memoria.evaluation import compare, evaluate
from memoria.experiments import execute
from memoria.scenarios import conditions, drifting_facts

store = ArtifactStore("var/artifacts")
dataset = store.put_record(drifting_facts(seed=7, keys=8, changes=8, probes_per_key=12))
runs = conditions(dataset, "statement-v1")  # baseline, drop, delay, reorder, ...

baseline = evaluate(execute(runs["baseline"], store).digest, store)
treated = evaluate(execute(runs["contaminate"], store).digest, store)
comparison = compare(baseline.digest, treated.digest, store)  # one variable changed

m = comparison.measure("known.correct")
print(f"{m.baseline.estimate:.2f} -> {m.treatment.estimate:.2f} (n={len(m.pairs)})")
print(f"difference {m.difference:+.3f}, 95% CI [{m.low:.3f}, {m.high:.3f}], p={m.p_value:.2g}")
for t in comparison.transitions:  # which probes failed, and how
    print(t.baseline.value, "->", t.treatment.value, len(t.probes))
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
