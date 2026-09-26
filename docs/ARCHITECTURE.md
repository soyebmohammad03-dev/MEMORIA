# MEMORIA Constitution & Architecture

Status: **authoritative**. Changes to this document are deliberate, reviewed, and recorded
in the Decision Log (§9). Code that contradicts it is a bug in one of the two.

## 1. Purpose

MEMORIA is an experimental observatory for one question:

> What actually happens to an AI system's memory as experiences accumulate, change,
> conflict, and disappear over time?

It is an instrument, not a product. Its outputs are **measurements and evidence**, not
chat features. A memory system under study is a *subject*; MEMORIA is the apparatus
that feeds it experiences, intervenes, observes, and records.

## 2. Research Principles

1. **Evidence or silence.** No claim, metric, or chart without the run that produced it.
   Missing evidence is reported as missing, never filled in.
2. **Reproducibility is a requirement.** Every run is determined by a manifest: code
   version, config hash, dataset hash, seeds, model identifiers. Same manifest → same
   result, or the nondeterminism is identified and recorded.
3. **Uncertainty is explicit.** Metrics carry sample size and an interval estimate.
   Point estimates alone are not results.
4. **Controlled comparison.** Effects are claimed only against a baseline under the same
   manifest with one variable changed.
5. **Failures are data.** Failed runs, negative results, and anomalies are retained and
   reportable, not discarded.
6. **Separate observation from interpretation.** Raw records are stored apart from the
   derived analyses that interpret them.

## 3. Engineering Invariants

These are enforced in code and tests, not by convention.

| # | Invariant |
|---|-----------|
| I1 | Records are immutable once committed. Change = new version, never mutation. |
| I2 | Every record is content-addressed (`sha256:` over canonical JSON). |
| I3 | Memory version history is an append-only, hash-linked chain (`supersedes`). |
| I4 | Every created memory cites ≥1 source `Experience` (no provenance-free memory). |
| I5 | All timestamps are timezone-aware and normalised to UTC. |
| I6 | Time is bitemporal: *valid time* (true in the world) ≠ *record time* (known to system). |
| I7 | Forgetting is a tombstone operation; history is never deleted by the system. A tombstone ends its chain. |
| I8 | Schemas are closed (`extra="forbid"`): unknown fields are errors, not silently kept. |
| I9 | Stochastic components receive an explicit seed; no hidden global RNG state. |
| I10 | Artifacts are written once, addressed by hash, and referenced from a manifest. |
| I11 | Record time never goes backwards, within a chain or across the log. |
| I12 | A version may only cite experiences already in the log that occurred at or before its `recorded_at`. |
| I13 | Stored records are re-verified against their digest on every read; mismatch is an error, never repaired silently. |

Tooling standard: Python ≥3.12, Pydantic v2, `mypy --strict`, Ruff, pytest with warnings
as errors, `uv` with a committed lockfile, CI on every push.

## 4. Core Domain Concepts

| Concept | Meaning | Status |
|---|---|---|
| **Experience** | A raw source interaction exactly as observed. Root of all provenance. | `core.py` |
| **MemoryVersion** | One immutable version of a memory; bitemporal; hash-linked to its predecessor. | `core.py` |
| **Operation** | Why a version exists: `create`, `update` (world changed), `correct` (prior was wrong), `forget` (tombstone). | `core.py` |
| **Belief** | A version plus the validity interval it effectively holds after later operations (§4.1). | `core.py` |
| **Memory state** | The versions believed at record time *k* to hold at valid time *t*. A pure fold over the log, with its own digest. | `core.py`, `store.py` |
| **Retrieval trace** | Query, backend, candidates, scores, and selected version digests. | Phase 2 |
| **Response** | Model output plus the retrieval trace and prompt that produced it. | Phase 2 |
| **Intervention** | A controlled perturbation: inject contradiction, contaminate, delete, delay, reorder. | Phase 3 |
| **Run manifest** | Everything that determines a run (see Principle 2). Its hash is the run ID. | Phase 3 |
| **Measurement** | A metric value with n, interval, and the run/artifact digests it came from. | Phase 4 |
| **Autopsy** | The provenance DAG from a response back through retrieval → versions → operations → experiences, at a point in both time axes. | Phase 5 |

The `update`/`correct` distinction is deliberate: it separates *temporal change* from
*error repair*, which is necessary to study supersession and contradiction.

### 4.1 History semantics

A memory's chain is folded into an ordered, non-overlapping belief timeline
(`core.beliefs`), considering only versions with `recorded_at <= k`:

| Operation | Effect on the timeline | Constraint |
|---|---|---|
| `create` | Starts the timeline with its own interval. | Version 1 only; cites ≥1 experience. |
| `update` | Appends; the previous belief ends where this one begins. | `valid_from` after the previous belief's. |
| `correct` | Retracts the latest belief entirely, then appends (may re-date). | `valid_from` after the belief that remains before it, if any. |
| `forget` | Retracts every belief. The chain is terminal. | Carries no content and no validity interval. |

Each belief's effective end is the earlier of its own `valid_to` and the next belief's
`valid_from`. State at (*t*, *k*) holds, per memory, the one belief (if any) valid at *t*.

Known limits: `correct` targets only the latest belief, and re-learning a forgotten fact
creates a new memory rather than reviving the tombstoned one.

## 5. Module Boundaries

The dependency rule is strict: **dependencies point inward toward `core`**. `core` imports
nothing from MEMORIA. Packages are created when their phase starts, not before.

```
core          records, hashing, time, history semantics, state fold  (exists)
store         append-only operation log; SQLite first, Postgres-capable  (exists)
formation     experience → memory operations (rule-based baseline, pluggable LLM extractors)
retrieval     lexical + embedding retrieval, retrieval traces
providers     adapters for LLMs, embedders, vector indexes (local-first)
experiments   manifests, datasets, interventions, deterministic runner, artifact store
evaluation    metrics, statistical tests, failure taxonomy
provenance    autopsy / provenance graph (NetworkX)
api           FastAPI surface over the above (no logic of its own)
observatory   interactive visualisation (consumes api only)
reports       evidence-backed reports generated from stored runs
```

### Extension points

Pluggability is achieved with `typing.Protocol` interfaces owned by the consuming module,
introduced **when the first implementation is written** (not as empty scaffolding):

- `MemoryBackend` — the subject under study (store + formation + retrieval policy)
- `Embedder`, `VectorIndex`, `LanguageModel` — provider adapters
- `Dataset`, `Intervention`, `Metric` — experiment components

Adding a backend, model, dataset, or metric must require **no change to `core` or the
runner**, only a new implementation and its registration in a manifest.

## 6. Storage

- **Operation log** (source of truth, `store.MemoryLog`): append-only SQLite tables of
  `Experience` and `MemoryVersion` records, stored as canonical JSON keyed by digest.
  Triggers reject every `UPDATE`/`DELETE`; `PRAGMA user_version` pins the schema.
  Every append is validated by `core` inside one write transaction; `verify()` re-checks
  the whole log. All domain rules live in `core`, so a PostgreSQL log must reproduce only
  storage and append-only enforcement (triggers/permissions there are backend-specific).
- **Derived indexes** (vector index, graph, caches) are rebuildable from the log and are
  never authoritative.
- **Artifacts** live under a local, git-ignored `var/` directory, content-addressed, and
  referenced from run manifests.

## 7. Experimental Lifecycle

```
experiences → formation → memory state → retrieval → response → provenance
    → intervention → measurement → failure discovery → statistics → replay → report
```

Every arrow is a recorded, replayable transformation.

## 8. Roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 ✓ | Foundation: constitution, tooling, core records | CI green; invariants tested |
| 1 ✓ | Operation log on SQLite; bitemporal state reconstruction ("state as of t") | Replay reproduces state bit-for-bit from the log |
| 2 | Formation baseline + retrieval (lexical, local embeddings); retrieval traces | A response is traceable to exact version digests |
| 3 | Experiment runner: manifests, seeds, datasets, interventions, artifact store | Re-running a manifest reproduces its artifacts |
| 4 | Evaluation: metrics, interval estimates, paired tests, failure taxonomy | Baseline vs. intervention comparison with CIs |
| 5 | Provenance graph & memory autopsy | Full autopsy of any stored response |
| 6 | API + observatory UI | UI renders only stored evidence |
| 7 | Report generation | Reports cite run/artifact digests for every claim |

## 9. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-26 | Memory history is event-sourced (append-only log; state is a fold). | Replay, reconstruction, and autopsy become native rather than bolted on. |
| 2026-09-26 | Bitemporal timestamps on every memory version. | Needed to distinguish "what was true" from "what the system believed, when". |
| 2026-09-26 | Content addressing via SHA-256 over canonical JSON. | Stable identity for provenance links and artifact dedup; stdlib only. |
| 2026-09-26 | Protocols introduced with first implementation, packages with their phase. | Avoid speculative abstractions; the boundaries above are the contract. |
| 2026-09-26 | `forget` carries no content or validity interval. | A tombstone asserts nothing; requiring those fields would force fabricated values. |
| 2026-09-26 | The state fold lives in `core`, not `store`. | It is pure domain logic; any backend reuses it, and it is testable without I/O. |
| 2026-09-26 | Record time is supplied by the caller, never read from a clock by the log. | Keeps appends and replays deterministic. |
| 2026-09-26 | `derived_from` is stored as a sorted, de-duplicated set. | Same citations must yield the same digest. |
