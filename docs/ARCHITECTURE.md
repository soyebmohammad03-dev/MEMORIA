# MEMORIA Constitution & Architecture

Status: **authoritative**. Changes to this document are deliberate, reviewed, and recorded
in the Decision Log (§9). Code that contradicts it is a bug in one of the two.

## 1. Purpose

MEMORIA is an experimental laboratory and observatory for studying the formation,
representation, consolidation, retrieval, revision, forgetting, interference,
contamination, provenance, and reliability of artificial memory over long interaction
horizons. Its founding question remains:

> What actually happens to an AI system's memory as experiences accumulate, change,
> conflict, and disappear over time?

It is an instrument, not a product. Its outputs are **measurements and evidence**, not
chat features. A memory system under study is a *subject*; MEMORIA is the apparatus
that feeds it experiences, intervenes, observes, and records. Different memory
architectures and policies are compared under controlled, reproducible conditions.

### 1.1 The research loop

```
experience → encoding → memory formation → consolidation → memory representation
  → indexing → retrieval → response → evaluation → feedback → belief revision
  → forgetting / interference → longitudinal analysis
```

Every arrow is a recorded, replayable transformation with provenance. Phases 1–4
implement experience → formation → representation (bitemporal versions) → retrieval →
response → evaluation; the remaining arrows are the roadmap (§8).

### 1.2 Research questions

Each phase names the questions it serves (see [ROADMAP.md](ROADMAP.md)). Answers are
claims only when backed by stored runs and statistics (Principles 1, 3, 4).

| # | Question |
|---|---|
| RQ1 | What information becomes durable memory? |
| RQ2 | What causes memories to be forgotten? |
| RQ3 | When do semantically similar memories interfere? |
| RQ4 | How do contradictory experiences change stored beliefs? |
| RQ5 | How does source reliability affect memory? |
| RQ6 | Does repeated exposure strengthen false information? |
| RQ7 | How does temporal distance affect retrieval? |
| RQ8 | Does semantic abstraction improve or damage factual fidelity? |
| RQ9 | How does consolidation trade specificity for generalisation? |
| RQ10 | How resilient is a memory system to contamination? |
| RQ11 | Can an incorrect memory be corrected without damaging unrelated memories? |
| RQ12 | How does retrieval policy change long-horizon reliability? |
| RQ13 | How does memory capacity affect forgetting and interference? |
| RQ14 | How does retrieval frequency change future recall? |
| RQ15 | Can the system explain exactly why a particular memory influenced a response? |

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
7. **Models are experimental artifacts.** Every model (embedder, reranker, language
   model) is identified by name, version, configuration and pinned weights, and its
   outputs are stored and verifiable like any other artifact.
8. **Honest labels.** Nothing is called semantic, neural, learned or "AI" unless it is.
   Deterministic heuristics are named as what they are.
9. **Laptop-feasible and local-first.** The deterministic core runs without a GPU,
   network access or heavy dependencies. Neural components are optional, local, small,
   and pluggable; no provider is hard-coded.

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
| I14 | A stored retrieval freezes the past: later versions and decisions must be recorded strictly after its `known_at`, so the state it observed can always be recomputed. |
| I15 | Every formation outcome is recorded, including decisions to change nothing, with the policy, a reason from a closed vocabulary, and the version it was judged against. |
| I16 | A retrieval trace lists every memory of the queried state and re-derives its own eligibility, totals, ranking and selection; the state it names must be the log's state at its coordinates. |
| I17 | A response cites only versions its trace selected, and cites evidence iff it answers. |
| I18 | Derived scores are quantised (12 decimal places) before recording, so digests do not depend on last-bit floating-point differences between platforms. |
| I19 | Interventions transform datasets (the input stream), never a memory log. Stored history is not branched, edited or replayed selectively. |
| I20 | Probes and their expectations are never changed by interventions: the instrument stays fixed while the input is perturbed. |
| I21 | Every seeded choice is `unit_interval(seed, *labels)`: stateless, platform-independent, and a function of content rather than position or call order. |
| I22 | A run's record and every artifact it names are a deterministic function of its manifest and input artifacts. Environment details are recorded separately and never enter those digests. |
| I23 | Manifest names are resolved through a registry and checked by round-trip: a resolved component must describe itself exactly as the manifest does. |
| I24 | Evaluation only reads run artifacts and writes new ones (`Evaluation`, `RunComparison`). Ground truth and run outputs are never altered. |
| I25 | Every measurement lists the probe ids in its numerator and denominator, and its estimate and interval are validated to follow from those counts. |
| I26 | Undefined is not zero: n = 0 gives no estimate and no interval. Unscorable outputs are excluded from accuracy denominators and reported as their own measurement. |
| I27 | Effects are measured only by paired comparison on identical probes of two runs that differ in exactly one manifest variable (Principle 4). |
| I28 | A classification records the rule that fired and the claims it relied on. Answer text decides *whether* an answer matches; provenance decides *why* it failed. |
| I29 | Every paired test reports its counts, a Newcombe interval, an exact p-value, its Holm adjustment within a family, and whether it is underpowered. |
| I30 | An embedder's identity is its `EmbedderSpec` digest (name, version, dimensions, preprocessing, normalisation, parameters). Every vector is validated for count, dimensionality, finiteness and declared normalisation. |
| I31 | Vectors from different embedder specs are never compared: loading, searching or verifying an index with another embedder is refused. |
| I32 | A semantic index binds its embedder spec, source state, entries, preprocessed-text digests and vector artifact by digest; rebuilding from the source state reproduces the vectors (bit-for-bit for deterministic embedders). |
| I33 | Manifest schema evolution never changes the digest of an existing manifest (§4.7). |

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
| **Formation decision** | The recorded outcome of one experience under one policy: reason, target memory, basis version, appended version (if any). | `core.py` |
| **Query** | Retrieval text pinned to both time axes (`valid_at`, `known_at`) and a result limit. | `core.py` |
| **Retrieval trace** | Query, full retriever spec, state digest, every candidate with per-signal scores and raw inputs, and the selected version digests. | `core.py` |
| **Response** | An answer (or abstention), the responder that produced it, and the exact version digests it cites. A prompt is added when an LLM responder exists. | `core.py` |
| **Step** | An experience plus the record time at which a system under study ingests it. | `core.py` |
| **Probe / Expectation** | A question pinned to (valid, known) time, with epistemic ground truth: `known` (one value), `unknown`, or `contested` (several values, undecided). | `core.py` |
| **Dataset** | A named, versioned, ingestion-ordered stream of steps plus probes. | `core.py`, `scenarios.py` |
| **Intervention** | A controlled, seeded perturbation of a dataset: delete (`drop`), `delay`, `reorder`, `contaminate`, `inject` (e.g. contradictions). Recorded as an `InterventionRecord` (spec, input, output, step diff). | `interventions.py`, `core.py` |
| **Run manifest** | Everything that determines a run (see Principle 2). Its hash is the run ID. | `core.py` |
| **Run record / outcomes** | Digests of every artifact a run produced; per-step decisions or rejections and per-probe answers next to expectations. | `core.py`, `experiments.py` |
| **Execution** | The environment a run was executed in. Evidence about a run, not part of its identity. | `experiments.py` |
| **Reading / Agreement** | What an output asserts (reader, original text, normalised tokens) and whether that agrees with the expectation. | `comparison.py` |
| **Claim / Relation** | A statement experience read as a claim about one key; how two claims relate (independent, same, retraction, contradiction, correction, temporal change). | `taxonomy.py` |
| **Classification** | A probe outcome class, the rule that fired, the failure locus, and the claims used. | `taxonomy.py` |
| **Measurement** | A metric value with n, interval, the probes counted, and (via its `Evaluation`) the run it came from. | `evaluation.py`, `statistics.py` |
| **Evaluation / RunComparison** | All probe judgements and measurements for one run; a paired baseline-vs-treatment comparison of two evaluations. | `evaluation.py` |
| **Embedder / EmbedderSpec** | Text → fixed-dimension vectors under a recorded identity; `HashedNgramEmbedder` is the deterministic lexical-subword reference (not semantic). | `embeddings.py` |
| **Semantic index** | Embeddings of a memory state as a stored, verifiable artifact with exact cosine search. | `semantic.py` |
| **Autopsy** | The provenance DAG from a response back through retrieval → versions → operations → experiences, at a point in both time axes. | Phase 17 |

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

### 4.2 Formation semantics

`formation.form(log, experience, policy, recorded_at=k)` atomically appends the
experience, asks the policy for a decision, appends the resulting version (if any), and
appends the decision. An experience already in the log yields `duplicate_experience`
without consulting the policy. A log is formed by exactly one policy.

Policies are pure functions of (experience, read-only history, `k`). Two baselines exist:

- **`episodic-v1`** stores each experience verbatim as memory `episode:<digest>`, valid
  from when it occurred. Nothing is superseded; contradictions coexist.
- **`statement-v1`** maintains one memory per key from an explicit statement language —
  `set <key> = <value>`, `correct <key> = <value>`, `forget <key>` (lowercase verbs;
  keys `[a-z0-9_.-]+`; one statement per experience). It does not interpret natural
  language. Memory content is `<key> = <value>`. For a statement occurring at *t* against
  the live memory's current belief `head`:

| Condition (checked in order) | Reason | Operation |
|---|---|---|
| not a statement | `unparsed` | — |
| no live memory, `set` | `new_key`, or `relearned` (new memory `key#n`, basis = tombstone) | create, valid from *t* |
| no live memory, `correct`/`forget` | `unknown_key` | — |
| *t* < `head.valid_from` | `stale` | — |
| `forget` | `explicit_forget` | forget |
| same content as `head` | `redundant` | — |
| *t* = `head.valid_from` | `conflicting` (first recorded wins) | — |
| `set` | `value_changed` | update, valid from *t* |
| `correct` | `explicit_correction` | correct, over `head`'s interval |

Known limits: late-arriving older facts are recorded as `stale` rather than inserted into
the past; same-instant conflicts are recorded, not resolved.

### 4.3 Retrieval semantics

A `Retriever` ranks the memories of `state_as_of(query.valid_at, query.known_at)`. Each
**signal** scores every memory and records named raw inputs. A candidate is eligible iff
the **gate** signal scores > 0; its total is Σ weight × score; candidates are ordered by
(eligible first, total descending, `memory_id` ascending) and the first `limit` eligible
are selected. Signals:

- **`bm25`**: Okapi BM25 (k1 = 1.2, b = 0.75 by default) over case-folded Unicode word
  tokens, no stemming or stop words, idf = ln(1 + (N − df + 0.5)/(df + 0.5)) with corpus
  statistics from the queried state; query terms de-duplicated. Inputs: per matched term
  tf, df, idf and contribution; document length, average length, N.
- **`recency`**: 0.5^(age/half-life) in days, on the record axis (age since recorded, as
  of `known_at`) or the valid axis (age since the belief began, as of `valid_at`).

Presets: `lexical()` = `bm25-v1` (BM25 only); `lexical_recency()` = `bm25+recency-v1`
(BM25 gate, recency weight 0.5, half-life 30 days). Weights are recorded in every trace;
they are baselines, not tuned values. The `extractive-top1-v1` responder answers with
the top selected memory's content verbatim and abstains when nothing is selected.

Known limits: tie-breaking by `memory_id` is deterministic but semantically arbitrary
(the episodic baseline demonstrably returns a stale report on a three-way tie); no
dense/semantic signal enters ranking yet (Phase 6; embeddings exist since Phase 5, §4.6).

### 4.4 Experiment semantics

`experiments.execute(manifest, store)`:

1. Resolves the policy, retriever, responder and interventions the manifest names (I23),
   and loads the dataset by digest, before doing any work. Unresolvable manifests fail
   without writing anything.
2. Applies interventions in order; stores every derived dataset and an
   `InterventionRecord` linking input → output with the exact step diff.
3. Ingests the effective dataset into a fresh log with `formation.form`. A step the log
   refuses is recorded as a rejection (with the reason) and the run continues.
4. Asks every probe with `retrieval.answer`, recording trace and response in the log.
5. Stores the log's canonical export, a `RunOutcomes` table, the manifest, the
   `RunRecord`, and a separate `Execution` record.

`reproduce(run, store)` re-executes the stored manifest and compares every artifact
digest; differing artifacts are kept alongside the originals.

**Interventions** (all parameters, including seeds, are recorded):

| Name | Effect | Never |
|---|---|---|
| `drop(rate, seed)` | removes selected steps | — |
| `delay(rate, seed, days)` | ingests selected steps later; the experience is unchanged | moves ingestion earlier |
| `reorder(seed, window_days)` | jitters ingestion order within a window, then keeps record time monotone | ingests anything earlier than planned |
| `contaminate(rate, seed, delay_days, source)` | adds a false copy (`<value> (contaminated)`) of selected `set`/`correct` statements; source `<source>:<original digest>`. `delay_days = 0` is a same-instant contradiction | touches non-statements |
| `inject(dataset)` | merges another stored dataset's steps (hand-built contradictions or contamination) | — |

**Ground truth** is epistemic and fixed on the authored dataset (I20): what an ideal
system could believe about `valid_at` from experiences recorded by `known_at` — the
latest report *by occurrence* holds; equal-time disagreement is `contested`; honouring
`forget` yields `unknown`. Recency of ingestion never makes a report more true. Under an
intervention the expectation is unchanged, so runs measure degradation against a fixed
reference. `RunOutcomes` records answers next to expectations; **judging them is Phase 4**.

Datasets: `scenarios.relocation_year()` (hand-built, 16 steps, 12 probes covering every
expectation status) and `scenarios.drifting_facts(seed, ...)` (generated, with ground truth
computed from the simulated world, `epistemic_truth`).

Not in Phase 3: contradiction/consistency *classification* (belongs with the Phase 4
failure taxonomy), scoring and statistics, a CLI.

### 4.5 Evaluation semantics

`evaluation.evaluate(run, store, spec)` reads a stored run and writes an `Evaluation`;
`evaluation.compare(baseline, treatment, store)` writes a `RunComparison`. The
`EvaluationSpec` (readers and their order, comparator `tokens-v1`, classifier
`provenance-v1`, confidence, alpha) is part of every result's identity.

Pipeline, per probe: observation (stored outcome, trace, response, log) → **reading**
→ **agreement** → **classification** → measurement → statistics.

**Comparison** (`comparison.py`). Values are equal iff their token sequences are equal
after Unicode NFKC, case folding and word tokenisation — no stemming, synonyms or fuzzy
thresholds. Readers, first applicable wins: `statement` (`set|correct k = v`),
`assignment` (`k = v`, memory content), `mention` (whole-token occurrences of dataset
values in free text; longer matches absorb contained ones; two distinct values =
ambiguous). The original output, reader and tokens are kept.

**Fact model** (`taxonomy.py`). Statement experiences are claims. Two claims on one key
are related by *occurrence* time: same instant and different values = contradiction;
different times = temporal change, or correction when the later one is a `correct`.
Ingestion order never decides. The *observed* claim is found through provenance
(cited version → source experience → claim); the *support* is the latest-occurring
authored claim(s) visible at `known_at` that assert an expected value.

**Outcomes** (one per probe; rules checked in this order, each recorded by id):

| Situation | Outcome | Category |
|---|---|---|
| abstained; expected `unknown` | `correct_abstention` | success |
| abstained; `contested` | `contested_abstained` | neutral |
| abstained; expected memory held / not held | `retrieval_miss` / `missing_memory` | failure |
| matched; `known` / `contested` | `correct` / `contested_answered` | success / neutral |
| several values mentioned | `ambiguous_output` | unscorable |
| unreadable; cites only memories about other subjects | `wrong_memory` | failure |
| unreadable; cites a subject memory asserting no value (`forget`) | treated as a non-answer (the abstention outcomes above) | — |
| unreadable otherwise | `malformed_output` | unscorable |
| mismatch; expectation has no supporting claim | `unclassifiable` | unscorable |
| answer not in the cited evidence / asserted by no visible claim | `unsupported` | failure |
| observed claim is about another key | `wrong_memory` (interference) | failure |
| observed claim was introduced by an intervention | `contaminated` | failure |
| `unknown` expected; claim later forgotten | `forgotten_recalled` | failure |
| claim not yet true at `valid_at` (corrections exempt: retroactive) | `temporal_error` | failure |
| claim is the value a correction replaced | `correction_failure` | failure |
| claim contradicts the support at the same instant | `contradiction` | failure |
| claim is older than the support | `stale` | failure |
| claim is newer than the support | `unsupported` | failure |

**Locus** of a failure: `retrieval` if a memory asserting the expected value was in the
queried state (or the answer came from another subject's memory), else `memory`. For
known-value failures without the expected memory, the **cause** is determined from the
run: `not_ingested` (never in the stream), `not_yet_ingested` (after `known_at`),
`rejected_by_log`, `rejected_by_policy` (with the policy's reason), `superseded`, or
`outside_validity` (believed, but its interval excludes `valid_at`).

**Measurements** (fixed set; each lists its probes): `coverage`, `unscorable` (all
probes); `known.correct`, `known.abstained`, `known.expected_in_state` (scorable
known-value probes); `known.expected_selected_given_in_state`; `unknown.abstained`;
`contested.answered`; `outcome.<class>` for every class (all probes; a partition);
`failure.locus.<locus>` (failures); `failure.cause.<cause>` (known-value failures without
the expected memory — causes may sum to less than the population if undetermined).

**Statistics** (`statistics.py`). Proportions: Wilson score interval. Paired comparison
of an indicator on the same probes: difference treatment − baseline with Newcombe (1998)
method 10 interval (continuity-corrected phi; reproduces the paper's Table III), exact
two-sided McNemar p-value, Holm adjustment within a family (`primary`, `outcome`,
`locus`), and `underpowered` when even the most one-sided split of the discordant pairs
could not reach `alpha` (e.g. ≤ 5 discordant pairs at 0.05). Transitions list which
probes moved between which outcomes.

**Assumptions and limits.** Intervals treat probes as independent; probes about the same
key share memory state, so intervals may be optimistic for clustered probes (cluster-aware
methods are deferred). Multiplicity is adjusted within a comparison, not across
comparisons. The fact model is the statement language: datasets without it are compared
on text, but claim-based classes cannot apply. The mention reader recognises only values
asserted somewhere in the dataset.

### 4.6 Semantic memory semantics (Phase 5, slice 1)

`embeddings.embed(embedder, texts)` applies the spec's `Preprocessing` (NFKC, case
folding, whitespace collapse, optional truncation — each recorded), encodes, and
validates. `HashedNgramEmbedder` hashes character n-grams of the preprocessed text with
SHA-256 into signed buckets and L2-normalises the integer counts; empty text is the zero
vector. It captures spelling, not meaning.

`semantic.SemanticIndex.build(state, embedder, store)` stores the `MemoryState` and a
little-endian float32 vector blob, and an `IndexManifest`. `search(text, k, embedder)`
returns `Neighbor`s ranked by quantised cosine over the stored float32 vectors, ties by
`memory_id` then version digest; zero vectors have similarity 0. `load` refuses a
different embedder; `verify` re-embeds the source state and requires identical texts
and bytes. Search is exact (brute force); approximate indexes must be validated against
it. Semantic signals do not yet enter retrieval ranking (Phase 6).

### 4.7 Experiment degrees of freedom and manifest evolution

The target experiment model varies, independently and explicitly: memory architecture,
formation policy, consolidation policy, retrieval policy, forgetting policy, source
model, embedding model, reranker, dataset, interventions, and evaluation specification.
Today's `RunManifest` records dataset, interventions, policy, retriever and responder;
evaluation specs are recorded by `Evaluation`.

Rule for adding degrees of freedom (I33): a new manifest field must have a default that
reproduces current behaviour, and a field at its default is omitted from the canonical
form, so every existing manifest keeps its digest and every existing run keeps its ID.
This "manifest schema v2" is implemented with the first phase that adds a degree of
freedom to runs (the embedding model, completing Phase 5), not before. Nothing that
changes results may live in global configuration.

## 5. Module Boundaries

The dependency rule is strict: **dependencies point inward toward `core`**. `core` imports
nothing from MEMORIA. Packages are created when their phase starts, not before.

```
core          records, hashing, time, history semantics, state fold  (exists)
store         append-only operation log; SQLite first, Postgres-capable  (exists)
formation     experience → recorded decisions and versions; episodic + statement baselines  (exists)
retrieval     signals, ranking, traces, extractive responder; lexical + recency  (exists)
scenarios     deterministic research datasets with probes and ground truth  (exists)
interventions seeded dataset perturbations  (exists)
artifacts     content-addressed write-once local store  (exists)
providers     adapters for LLMs, embedders, vector indexes (local-first)
experiments   registry, deterministic runner, reproduction check  (exists)
comparison    answer readings and agreement  (exists)
taxonomy      claims, relations, outcome classification  (exists)
statistics    Wilson, Newcombe paired, exact McNemar, Holm  (exists)
evaluation    evaluations, measurements, paired comparisons  (exists)
embeddings    embedding contract, preprocessing, reference embedder  (exists)
semantic      semantic index artifacts, exact search  (exists)
provenance    autopsy / provenance graph (NetworkX)
api           FastAPI surface over the above (no logic of its own)
observatory   interactive visualisation (consumes api only)
reports       evidence-backed reports generated from stored runs
```

### 5.1 Target architecture

Fifteen composable layers. Each builds on the Phase 1–4 substrate and must preserve its
invariants: history stays append-only and bitemporal, every derived object cites its
sources, and every experiment reproduces from its manifest.

| Layer | Responsibility | Builds on | Phase |
|---|---|---|---|
| L1 Memory core | Typed memories: episodic and semantic memories, temporal facts, entities, relations, sources, confidence, validity, lineage, consolidation state, supersession, contradiction and abstraction relationships — as explicit types, not a universal object | `core` versions and history | 1 ✓, extended 7, 8, 11, 12 |
| L2 Formation | Pluggable policies (verbatim, keyed, abstraction, entity-centric, relation extraction, summarisation, hybrid); experience → decision → memory → evidence | `formation` | 2 ✓, extended 7 |
| L3 Consolidation | candidate → validation → deduplication → merging → abstraction → durable memory, with lineage to every supporting experience | L1, L2, L5 | 7 |
| L4 Hybrid retrieval | Decomposed signals (lexical, semantic, temporal, recency, reliability, confidence, provenance, contradiction, diversity), rerankers, adaptive policies | `retrieval` signals and traces | 2 ✓, 6, 13 |
| L5 Semantic memory | Local embedders and indexes as artifacts | `artifacts`, `core` states | 5 (slice 1 ✓) |
| L6 Memory graph | Provenance and semantic graph with snapshots | all records | 8 |
| L7 Forgetting lab | Forgetting mechanisms as policies and interventions; nothing deleted | history, interventions | 9 |
| L8 Interference lab | Controlled similarity, density and repetition | L4, L5, datasets | 10 |
| L9 Belief dynamics | Competing claims and revision | `taxonomy` relations | 4 ✓ (relations), 11 |
| L10 Source and trust | Source identity and reliability, independent of content | experiences | 12 |
| L11 Uncertainty | Separate memory, retrieval and response confidence; calibration | L4, evaluation | 12 |
| L12 Attack lab | Synthetic contamination and adversarial interventions | `interventions` | 3 ✓ (basic), 14 |
| L13 Benchmarks | Generated long-horizon datasets with ground truth | `scenarios` | 3 ✓ (seeded), 15 |
| L14 Autopsy | Response-to-experience explanation as a research object | L6, evaluation | 17 |
| L15 Longitudinal analysis and observatory | Curves, advanced statistics, API/UI over stored evidence | `evaluation`, `statistics` | 16, 19, 20 |

### Extension points

Pluggability is achieved with `typing.Protocol` interfaces owned by the consuming module,
introduced **when the first implementation is written** (not as empty scaffolding):

- `formation.FormationPolicy` — pure decision function (exists: `episodic-v1`, `statement-v1`)
- `retrieval.Signal` — per-memory scorer with named raw inputs (exists: `bm25`, `recency`).
  A dense/embedding signal plugs in here without changes to `core`.
- `MemoryBackend` — the subject under study (store + formation + retrieval policy)
- `Embedder`, `VectorIndex`, `LanguageModel` — provider adapters
- `interventions.Intervention` — dataset → dataset perturbation (exists: `drop`, `delay`,
  `reorder`, `contaminate`, `inject`)
- `experiments.Registry` — resolves manifest names; `extend()` adds components, never
  redefines them
- `embeddings.Embedder` — text → vectors under an `EmbedderSpec` (exists:
  `hashed-char-ngrams`; neural adapters behind an optional extra, pending)
- `Dataset` (a record, not a protocol: any generator producing one plugs in)
- Evaluation components are versioned names in `EvaluationSpec` (`tokens-v1`,
  `provenance-v1`, readers). One implementation each exists, so there is no registry;
  an unimplemented name is refused rather than silently substituted.

Adding a backend, model, dataset, or metric must require **no change to `core` or the
runner**, only a new implementation and its registration in a manifest.

## 6. Storage

- **Operation log** (source of truth, `store.MemoryLog`): append-only SQLite tables of
  `Experience` and `MemoryVersion` records, stored as canonical JSON keyed by digest.
  Triggers reject every `UPDATE`/`DELETE`; `PRAGMA user_version` pins the schema.
  Every append is validated by `core` inside one write transaction; `verify()` re-checks
  the whole log. All domain rules live in `core`, so a PostgreSQL log must reproduce only
  storage and append-only enforcement (triggers/permissions there are backend-specific).
- **Derived records** (schema v2): an append-only `records` table holds formation
  decisions, retrieval traces and responses, each validated against the records it
  references on append and again by `verify()`. v1 logs are migrated additively on open.
  Identical records are idempotent (same content address = same observation).
- **Derived indexes** (vector index, graph, caches) are rebuildable from the log and are
  never authoritative.
- **Semantic indexes**: an `IndexManifest` record, its source `MemoryState` record,
  and a raw vector blob (kind `IndexVectors`, little-endian float32, row-major), all in
  the artifact store.
- **Artifacts** (`artifacts.ArtifactStore`) live under a local, git-ignored `var/`
  directory: `objects/<hh>/<sha256 rest>` (read-only files, written atomically, re-hashed
  on every read, never overwritten) plus an append-only `index.jsonl` of (digest, kind)
  for discovery. A record artifact's bytes are its canonical JSON, so artifact digest =
  record digest. A memory log is stored as its canonical JSONL export
  (`MemoryLog.export()`), not as a SQLite file; `MemoryLog.load()` rebuilds it through
  the validated append path, rejecting non-canonical or invalid lines.

## 7. Experimental Lifecycle

The research loop (§1.1), executed by the experiment layer:

```
dataset → interventions → formation (→ consolidation) → memory state (→ index)
  → retrieval → response → evaluation → comparison → longitudinal analysis → autopsy
```

Every arrow is a recorded, replayable transformation; bracketed steps are roadmap.

## 8. Roadmap

Rebaselined 2026-09-27 (§9). Phase detail — objective, capabilities, components,
research questions, experiments, artifacts, validation criteria, dependencies and
deferrals — is in [ROADMAP.md](ROADMAP.md).

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 ✓ | Foundation: constitution, tooling, core records | CI green; invariants tested |
| 1 ✓ | Operation log on SQLite; bitemporal state reconstruction | Replay reproduces state bit-for-bit from the log |
| 2 ✓ | Formation baselines, retrieval (lexical, recency), traces | A response is traceable to exact version digests |
| 3 ✓ | Experiment runner: manifests, seeds, datasets, interventions, artifact store | Re-running a manifest reproduces its artifacts |
| 4 ✓ | Evaluation: metrics, interval estimates, paired tests, failure taxonomy | Baseline vs. intervention comparison with CIs |
| 5 ◐ | Semantic memory and local embedding infrastructure | Index rebuilds reproduce vectors; neighbours trace to versions; core runs without neural deps |
| 6 | Hybrid retrieval and explainable ranking | Rank reproducible from trace; policies compared with paired statistics |
| 7 | Memory consolidation and abstraction | Every consolidated memory has lineage to all supporting experiences |
| 8 | Provenance and semantic memory graph | Graph is a pure, reproducible function of stored artifacts |
| 9 | Forgetting laboratory | Every forgotten memory reconstructible; retention measured with intervals |
| 10 | Interference laboratory | Each generator parameter measurably moves its target property |
| 11 | Contradiction and belief revision | Revision measured without equating newer with truer; collateral damage measured |
| 12 | Source reliability and uncertainty | Separate, calibrated confidence components; no universal score |
| 13 | Adaptive retrieval policies | Every adaptation a replayable event |
| 14 | Memory contamination and adversarial experiments | Every attack-caused failure traces to an injected experience |
| 15 | Long-horizon benchmark generation | Datasets reproducible from parameters with measured difficulty |
| 16 | Large-scale longitudinal evaluation | Curves decomposable to per-probe observations |
| 17 | Memory autopsy | Every autopsy link resolves to a verified artifact |
| 18 | Experiment orchestration and parameter sweeps | Sweeps reproduce; resumption equals uninterrupted runs |
| 19 | Advanced statistical and research analysis | Methods reproduce reference values; coverage checked |
| 20 | Interactive research observatory and API | Every rendered number links to its artifact |
| 21 | Reproducibility and research packaging | A bundle reproduces on a clean machine |
| 22 | Final benchmark and scientific validation | Every reported answer reproducible from a published bundle |

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
| 2026-09-27 | Neural embedding retrieval is deferred beyond Phase 2; the `Signal` contract is where it plugs in. | Local models add torch-scale dependencies and model downloads, and cross-platform float drift would break digest reproducibility. The exit criterion does not need them. Revisit with a pinned model and a quantised-evidence plan. |
| 2026-09-27 | Formation decisions are first-class records, including no-change outcomes. | "Why is this not in memory?" must be answerable from stored evidence, not re-derived. |
| 2026-09-27 | The statement baseline uses an explicit statement language, not NL heuristics. | A baseline must be exactly specifiable; NL extraction belongs to pluggable (e.g. LLM) policies. |
| 2026-09-27 | Retrieval has no single opaque score: per-signal scores with raw inputs, a gate, weights and ranking rule are all recorded. | Later analysis must be able to attribute a ranking to its causes. |
| 2026-09-27 | Traces include every memory of the state, not only the selected ones. | Makes "why was X not returned" answerable. |
| 2026-09-27 | Traces freeze the past (I14) instead of pinning a log position. | Keeps traces meaningful in pure time coordinates and reproducible by recomputation. |
| 2026-09-27 | Scores are quantised to 12 decimal places; pipeline digests are pinned by a golden test. | Detects platform or semantic drift in stored results. |
| 2026-09-27 | Interventions perturb datasets, never logs (I19). Rejected: branching or editing stored history to simulate perturbations. | Stored history stays immutable; every perturbation is itself a recorded, diffable artifact. |
| 2026-09-27 | Seeded choices use SHA-256 `unit_interval` (I21). Rejected: `random.Random`. | Python does not guarantee `shuffle`/`sample` algorithms across versions, and a shared generator makes results depend on call order. |
| 2026-09-27 | Probe expectations are epistemic and fixed on the authored dataset (I20). Rejected: recomputing truth per perturbed stream; deriving truth from a reference policy. | Recomputing hides degradation; policy-derived truth is circular. "Newer" is not "more true": occurrence, not ingestion, orders reports. |
| 2026-09-27 | The log artifact is a canonical JSONL export. Rejected: storing the SQLite file. | SQLite's page layout is not a byte-level contract; the export is, and it round-trips through the validated append path. |
| 2026-09-27 | The environment is a separate `Execution` record (I22). Rejected: including it (or wall-clock time) in the manifest or run record. | The same manifest must reproduce the same artifacts on any machine; where it ran is evidence, not identity. |
| 2026-09-27 | Components are resolved by name with a round-trip check (I23). Rejected: pickled callables or code hashes in manifests. | Names with explicit versions are human-readable and stable; the round-trip check prevents a name silently meaning different parameters. |
| 2026-09-27 | Rejected steps are recorded and the run continues. | A refusal by the log is a measured outcome, not a crash. |
| 2026-09-27 | Runs record answers beside expectations but do not score them. | Matching answers to expectations (e.g. across content formats) is an evaluation decision for Phase 4. |
| 2026-09-27 | Answers are compared by exact token sequences with explicit readers. Rejected: edit-distance or embedding similarity thresholds; an LLM judge. | A threshold hides a decision in a number and drifts with models; exact rules are auditable and reproducible. |
| 2026-09-27 | Failure classes come from provenance (cited version → experience → claim), not from answer text alone. | Text says an answer is wrong; only provenance says whether it was stale, contaminated, corrected, or from another subject. |
| 2026-09-27 | Claims relate by occurrence time; ingestion order never decides truth. Rejected: "latest ingested wins". | Newer is not more true; a temporal change is not a contradiction. |
| 2026-09-27 | Contested ground truth is neutral, and a valueless (retraction) answer is judged as a non-answer. | A conflict is not automatically an error; abstaining on insufficient evidence is not a failure. |
| 2026-09-27 | Wilson intervals for proportions. Rejected: Wald (leaves [0,1], poor at small n); Clopper-Pearson (overly conservative). | Good coverage at the small n typical here, closed form, deterministic. |
| 2026-09-27 | Newcombe method 10 for paired differences. Rejected: paired Wald; method 8 (coverage dips at small n); bootstrap (resampling, unstable at small n). | Recommended by Newcombe; closed form; verified against the paper's Table III. |
| 2026-09-27 | Exact McNemar with Holm within families, plus an underpowered flag. Rejected: chi-square McNemar (invalid for small discordant counts); Bonferroni (uniformly less powerful than Holm); unadjusted p-values. | Honest inference at small n: a result that cannot reach significance says so. |
| 2026-09-27 | Evaluation records live in `evaluation.py`, not `core.py`. | They are derived interpretation (Principle 6), kept apart from the recorded history they judge. |
| 2026-09-27 | Research scope rebaselined: a laboratory comparing memory architectures along the research loop (§1.1), with RQ1–RQ15 and layers L1–L15. Phases 1–4 are the substrate and are not changed. | Principles 7–9 added so the expansion cannot trade rigour for breadth. |
| 2026-09-27 | Old roadmap phases 5–7 (graph + autopsy, API/UI, reports) are superseded by phases 8, 17, 20, 19/21. | Autopsy needs the graph, sources, consolidation and longitudinal data first; a UI needs something worth rendering. |
| 2026-09-27 | Detailed roadmap lives in ROADMAP.md, bound by §8. | Keeps the constitution readable while every phase still has explicit objectives and validation criteria. |
| 2026-09-27 | Reference embedder is signed character-n-gram feature hashing, labelled lexical-subword. Rejected: random vectors (no structure to test); a bundled neural model (download, size, cross-platform float drift in the core suite). | Deterministic on every platform (integer counts, one correctly rounded division) and honest about what it measures. |
| 2026-09-27 | Vectors are stored as little-endian float32 in a separate artifact; search uses the stored values. Rejected: JSON floats (size); float64 (no benefit for comparison, larger); in-memory-only indexes (not auditable). | Compact, byte-exact, and the same results before and after reload. |
| 2026-09-27 | The first index is exact brute-force cosine. Rejected: FAISS/HNSW now. | An exact reference must exist before approximate indexes can be validated against it; laptop scale does not need ANN yet. |
| 2026-09-27 | Manifest fields are added only with defaults omitted from canonical form (I33), implemented with the first new degree of freedom. Rejected: adding optional fields now. | Adding fields naively would change every existing manifest digest and break Phase 3/4 reproducibility. |
| 2026-09-27 | No neural dependency added in this slice. | The contract, identity and index lifecycle are the prerequisite; a neural adapter without them would be unauditable. |
