# MEMORIA Research Roadmap

Status: **authoritative** companion to [ARCHITECTURE.md](ARCHITECTURE.md) (§8). Research
questions RQ1–RQ15 are defined in ARCHITECTURE.md §1.2; layers L1–L15 in §5.1.

Every phase must meet the constitution's quality bar: explicit semantics, typed
contracts, deterministic configuration recorded in manifests, provenance, reproducible
artifacts, tests including edge cases and adversarial cases, fixtures, measurable
outputs, documented assumptions and failure behaviour. A phase is complete only when
its validation criteria hold in CI.

Phases 1–4 are complete and form the **substrate**: bitemporal append-only history
(1), formation and retrieval with traces (2), reproducible experiments with
interventions (3), and evaluation with a failure taxonomy and paired statistics (4).
Phases 5–22 build on it without weakening any invariant. Phases 5 (semantic memory), 6
(hybrid, explainable retrieval), 7 (consolidation, abstraction, hierarchy) and 8–10 (memory
graph, forgetting and interference, built as one super-phase) and 11-12 (belief revision and
calibrated uncertainty, one laboratory) are complete. **Super-Phase 6** (autopsy, replay, counterfactuals and the multi-seed benchmark;
[results](experiments/superphase6-autopsy-replay-benchmark.md)) builds on it. **Super-Phase 5** (adaptive importance,
retention dynamics, free-text claims and entity identity;
[results](experiments/superphase5-adaptive-memory.md)) delivers the adaptive-importance and
feedback part of Phase 13 and the retention curves of Phase 16 on generated worlds; what remains
of those phases is listed in their sections.

## Release status (v0.1.0)

| Status | Phases |
|---|---|
| Complete | 1–4 (substrate), 5, 6, 7, 8–10, 11–12 |
| Partly delivered | 13 (adaptive importance and feedback, not adaptive policies inside hybrid retrieval), 16 (retention curves on generated worlds), 17 (autopsy of memory answers and beliefs, not of hybrid traces) |
| Not started | 14 (contamination and adversarial sources), 15 (parameterised benchmark generator), 18 (sweeps), 19 (advanced statistics), 20 (API and observatory), 21 (research bundles), 22 (final pre-registered study) |

The Super-Phase 6 benchmark is a multi-seed study over five fixed environments; it is not the
Phase 22 study. The scientific implementation is frozen at this release apart from defects.

---

## Phase 5 — Semantic memory and local embedding infrastructure (L5)

- **Objective:** make vector representations of memory reproducible experimental
  artifacts, so semantic retrieval can later be compared with lexical retrieval.
- **Capabilities:** embedding contract with recorded identity; deterministic
  dependency-free reference embedder; exact semantic index over a memory state with
  persistence, verification and compatibility checks; optional local neural embedders
  with pinned weights; approximate index checked against the exact one.
- **Components:** `embeddings` (`EmbedderSpec`, `Embedder`, `Preprocessing`,
  `HashedNgramEmbedder`), `semantic` (`IndexManifest`, `SemanticIndex`, `Neighbor`);
  later an optional `[neural]` extra with a sentence-embedding adapter and an optional
  ANN backend.
- **Questions:** prerequisite for RQ3 and RQ8 (similarity-driven interference;
  abstraction vs fidelity).
- **Experiments:** reproducibility of index builds (repeat, reload, second store);
  exact-vs-ANN agreement; neural cross-platform drift within declared tolerance.
- **Artifacts:** `IndexManifest`, vector blobs (little-endian float32), source
  `MemoryState`.
- **Validation:** rebuilding an index from its manifest reproduces its vectors
  bit-for-bit (reference embedder) or within a declared tolerance (neural, recorded in
  the spec); every neighbour traces to a memory version; the core suite runs without
  neural dependencies.
- **Dependencies:** Phases 1–4.
- **Deferred:** representations beyond memory content — entities, relations (Phases
  7–8). (Semantic signals inside ranking: done in Phase 6.)
- **Status:** complete. Reference embedder and exact index; `onnx-sentence` adapter with
  MiniLM-L6 pinned by revision and file hashes (optional `neural` extra); FAISS HNSW
  candidate generation with exact re-scoring (optional `ann` extra); manifest schema v2
  with `representation`; semantic signal in runs (exact index); the semantic diagnostic
  and the lexical-vs-neural study with results in docs/experiments/phase5-semantic.md;
  tolerance policy validated against the PyTorch reference. Open: cross-hardware
  equivalence of neural outputs (needs model download in CI or a second platform);
  numpy-accelerated exact search (not needed at current scale).

## Phase 6 — Hybrid retrieval and explainable ranking (L4)

- **Objective:** compare retrieval policies built from decomposed signals.
- **Capabilities:** semantic similarity, temporal relevance, source reliability,
  memory confidence, contradiction status and diversity as `Signal`s, each with raw
  inputs; rerankers as recorded pipeline stages; retrieval policy as a manifest variable.
- **Components:** `retrieval` signals; `RerankerSpec`; policy presets; trace extensions
  for multi-stage ranking.
- **Questions:** RQ7, RQ12 (temporal distance; retrieval policy vs long-horizon
  reliability).
- **Experiments:** lexical vs semantic vs hybrid on the same probes (paired, Phase 4);
  ablation of each signal.
- **Artifacts:** traces with per-stage, per-signal evidence.
- **Validation:** every selected memory's rank is reproducible from the trace alone;
  paired comparisons between policies run through `compare`.
- **Dependencies:** 5.
- **Deferred:** learned or adaptive weights (13); source and confidence signals' semantics (12).
- **Inputs from Phase 5:** semantic retrieval surfaces the right subject but not the right
  value (numeric changes and contradictions score like paraphrases); the reference and
  neural representations overlap in only 56% of top-5 results. Hybrid ranking should be
  evaluated on exactly these failure modes, with ANN as candidate generation.
- **Status:** complete. `memoria.hybrid` provides a corpus of every known version, three
  candidate generators (lexical, semantic via exact or HNSW index, metadata), typed hard
  filters, nine registered signals with explicit missing-value semantics, four versioned
  normalisations, weighted and rank-fusion scoring, MMR diversity, four contradiction
  exposure modes, and self-checking traces whose explanations and score decompositions are
  re-derived on load. `memoria.hybrid_eval` runs the ablation ladder, leave-one-out against
  two references, counterfactual perturbations, twelve adversarial cases and paired
  statistics on a hand-built benchmark, with separate performance records. Results:
  docs/experiments/phase6-hybrid.md. Validation as held: every selected memory's rank
  (and every score) is reproducible from the trace alone, and policies are compared with
  paired statistics from `memoria.statistics`. **Deviation:** the comparisons run inside
  the hybrid experiment on benchmark queries, not through `evaluation.compare`, because
  hybrid policies are not yet a run-manifest variable.
- **Deferred:** a run-manifest field for retrieval policies (**done in Phase 7**: manifest
  schema v3, `execute` and `compare` cover hybrid retrieval); claims for free-text memories (Phases 7–8); corrections that are retroactive at
  the memory level, not only the claim level (Phase 11); memory confidence as a signal
  (Phase 12); a larger generated benchmark (Phase 15); indexing every version for the
  semantic generator; learned or adaptive weights (Phase 13).

## Phase 7 — Memory consolidation and abstraction (L3, L1)

- **Objective:** study how transient experiences become durable memories.
- **Capabilities:** explicit pipeline candidate → validation → deduplication → merging →
  abstraction → consolidation → durable memory; typed episodic and semantic memories;
  consolidated memories citing all supporting experiences.
- **Components:** `consolidation` policies and records; memory types in core with
  lineage (`consolidated_from`); consolidation decisions as records.
- **Questions:** RQ1, RQ8, RQ9 (what becomes durable; abstraction vs fidelity;
  specificity vs generalisation).
- **Experiments:** consolidation on vs off; merge thresholds varied; fidelity of
  abstractions against ground truth.
- **Artifacts:** consolidation decisions; lineage per memory.
- **Validation:** no consolidated memory without lineage to every supporting
  experience; consolidation is replayable and reversible in analysis (history kept).
- **Dependencies:** 5, and 6 for similarity-based merging.
- **Deferred:** graph queries (8); summarisation by language models.
- **Status:** complete. `DerivedMemory` (core) with levels L0–L4 and epistemic status;
  `consolidation` (grouping regimes exact / canonical / claim / temporal / semantic,
  complete-linkage with recorded guard refusals, occurrence-ordered timelines with
  retroactive corrections and contested periods, L3 entity profiles, L4 timelines and
  inferred co-changes, decomposed importance, recency and importance promotion, structured
  loss reports, invalidation, deterministic replay); manifest schema v3 (retrieval and
  consolidation policies by digest; hybrid and consolidated runs through `execute`,
  `evaluate` and `compare`); level-aware hybrid retrieval (levels, `support` signal, stale
  exclusion, exact scan generator); `consolidation_eval` (strategies A–H, ablations,
  stability–plasticity sweeps, retrieval modes, demonstration) on eight generated worlds.
  Results: docs/experiments/phase7-consolidation.md.
- **Deferred after Phase 7:** access-frequency and feedback-driven importance (13);
  claims for free-text memories and entity resolution beyond surface capitals (8);
  language-model summaries as a consolidation regime, behind the same loss reports;
  cross-hardware reproduction of neural semantic dedup (the lab uses the deterministic
  reference embedder); incremental consolidation (each checkpoint rebuilds from history).

## Phase 8 — Provenance and semantic memory graph (L6)

- **Objective:** a deterministic graph over everything MEMORIA records.
- **Capabilities:** nodes for experiences, memories, versions, claims, entities,
  sources, retrievals, responses, experiments, interventions, evaluations; edges
  `created_by`, `supports`, `contradicts`, `supersedes`, `derived_from`,
  `retrieved_by`, `influenced`, `corrects`, `consolidated_from`, `related_to`;
  snapshots.
- **Components:** `graph` builder (NetworkX, optional dependency), snapshot records.
- **Questions:** RQ15 (explaining influence).
- **Experiments:** reconstruction equality; graph statistics across conditions.
- **Artifacts:** canonical graph exports and snapshot manifests.
- **Validation:** the graph is a pure function of stored artifacts; rebuilding yields an
  identical canonical export; every edge cites the record that justifies it.
- **Dependencies:** 1–7.
- **Deferred:** interactive exploration (20); graph learning.
- **Status:** complete (with 9 and 10). `memoria.graph`: typed nodes and provenance-aware
  edges (rule, evidence, epistemic status never stronger than an endpoint; only
  record-copying edges observed), structured claims only for structured evidence,
  timeline events, contradictions and supersession, derived claims linked to what they
  restate, inferred co-mention links, snapshots with source-state hashes, rebuild
  verification, diffs, decomposed analytics (all and active views), temporal diagnostics,
  bounded provenance traversal, and a retrieval view (`graph` generator, `graph_entity`,
  `graph_claim`, `graph_contradiction` signals). `memoria.entities`: conservative, recorded,
  reversible resolution. Manifest schema v4 (`graph_policy`, in the "retrieval" variable).
  **Deviations:** no NetworkX (standard library, canonical records); runs record snapshots
  by digest and rebuild them rather than storing them; node types for evaluations and
  interventions, and `created_by`/`influenced`/`corrects` edges, are not built (evaluations
  and interventions are reached through the run; corrections are `supersedes[correction]`).
  Results: docs/experiments/phase8-10-graph-forgetting-interference.md.

## Phase 9 — Forgetting laboratory (L7)

- **Objective:** make forgetting measurable instead of a deletion.
- **Capabilities:** intentional forgetting, time decay, interference-based,
  selective, retrieval-induced forgetting, capacity pressure, replacement,
  consolidation resistance — each an explicit policy or intervention.
- **Components:** `forgetting` policies producing recorded operations (never physical
  deletion); capacity models.
- **Questions:** RQ2, RQ13 (causes of forgetting; capacity).
- **Experiments:** retention under each mechanism; dose–response on decay and capacity.
- **Artifacts:** forgetting decisions; retention measurements.
- **Validation:** every forgotten memory is reconstructible from history; retention
  curves carry intervals.
- **Dependencies:** 3, 4, 7.
- **Deferred:** curve fitting (16).
- **Status:** complete. `memoria.forgetting`: availability states (active, suppressed,
  archived, excluded, forgotten), ten rules (age, fifo, recency with graded suppression,
  importance, access from earlier traces, validity, contradiction, provenance, hybrid votes,
  selective by entity / interval / source / kind / contradiction / level), cascade or
  explicit retention of derived memories with dangling and unsupported-abstraction
  diagnostics, preserved aggregates, and memory-level measurements (retention, precision,
  recall, accidental retention, collateral, provenance completeness, stale rate,
  contradiction visibility). Manifest schema v4 (`forgetting_policy`, its own variable);
  hard exclusion `forgotten_by_policy`, the `suppression` signal. **Deviations:**
  interference-based, retrieval-induced and capacity-pressure forgetting are represented
  only as the access, fifo and hybrid rules; retention curves over time are Phase 16.

## Phase 10 — Interference laboratory (L8)

- **Objective:** controlled study of retrieval competition.
- **Capabilities:** generators varying semantic, lexical, temporal and source
  similarity, contradiction, repetition, retrieval frequency and density.
- **Components:** interference dataset generators; competition measurements from traces.
- **Questions:** RQ3, RQ14.
- **Experiments:** similarity sweeps; density sweeps; per-condition failure shifts.
- **Artifacts:** generated datasets (content-addressed), competition statistics.
- **Validation:** each generator parameter measurably moves its target property; effects
  reported with paired intervals.
- **Dependencies:** 4, 5, 6.
- **Deferred:** sweep orchestration (18).
- **Status:** complete. `memoria.interference`: eight mechanisms (proactive, retroactive,
  temporal, semantic near-collision, entity, contradiction, consolidation, retrieval mix)
  with frequency, wording, recency and source modifiers; nested, fully labelled
  populations; manipulation checks per parameter; observations with target and distractor
  roles and three confusions; load curves against the load-0 baseline with paired
  statistics, Holm across loads, onset, and the interference / retrieval-failure
  decomposition. World-level interference is an `inject` intervention with four distractor
  kinds.

## Phase 11 — Contradiction and belief revision (L9)

**Status: complete** (with Phase 12, as one laboratory; see [results](experiments/phase11-12-beliefs-calibration.md)). Deviations: beliefs and their subsystems are new modules (`beliefs`, `revision`, `contradictions`) rather than an extension of `taxonomy.Relation`; the contradiction taxonomy is built on the Phase 8 graph snapshot.

- **Objective:** represent competing claims and how beliefs change.
- **Capabilities:** belief states over competing claims; compatible, temporal change,
  correction, direct contradiction, unresolved conflict, source-dependent disagreement;
  revision operators as recorded operations.
- **Components:** `beliefs` subsystem extending `taxonomy.Relation`; revision policies.
- **Questions:** RQ4, RQ6, RQ11.
- **Experiments:** revision policies under contradiction rates; correction without
  collateral damage to unrelated memories.
- **Artifacts:** belief-state snapshots; revision decisions.
- **Validation:** "newer" never implies "truer" by construction; unrelated-memory
  damage is measured, not assumed.
- **Dependencies:** 4, 8.
- **Deferred:** source weighting (12).

## Phase 12 — Source reliability and uncertainty (L10, L11)

**Status: complete** (see above). Deviation: confidence components are carried by beliefs and predictions, not attached to memory versions; adaptive use of confidence in retrieval stays deferred to Phase 13.

- **Objective:** make source and confidence explicit variables that can be varied
  independently of content.
- **Capabilities:** source identity, type and reliability metadata; provenance strength;
  supporting and conflicting evidence per memory; memory, retrieval and response
  confidence kept as separate components; evidence sufficiency; calibration evaluation.
- **Components:** `sources` model (a manifest component); confidence records attached to
  versions, traces and responses; calibration measurements in `evaluation`.
- **Questions:** RQ5, RQ6 (source reliability; repetition of false information).
- **Experiments:** identical content from sources of varied reliability; calibration of
  abstention and response confidence against Phase 4 outcomes.
- **Artifacts:** source models, confidence components, calibration tables with intervals.
- **Validation:** no universal confidence score; each component has a definition, its
  inputs, and a calibration measurement.
- **Dependencies:** 6, 11.
- **Deferred:** adaptive use of confidence in retrieval (13); adversarial sources (14).

## Phase 13 — Adaptive retrieval policies (L4)

**Status: partly delivered by Super-Phase 5.** Delivered (`adaptive`, `retention`): access and
feedback events in a leakage-safe ledger, an auditable, ablatable importance model, and
retention schedules driven by it, studied against static baselines with paired statistics.
Still open: adaptive *retrieval policies* inside the hybrid pipeline (signals reweighted by
feedback, policy state as replayable snapshots), a manifest variable for them, and retrieval-
induced strengthening measured on the Phase 6 corpus.

- **Objective:** study retrieval policies whose behaviour changes with use.
- **Capabilities:** retrieval-frequency effects, feedback-driven reweighting and
  retrieval-induced strengthening, each as a recorded state transition.
- **Components:** adaptive policy contract (state + update rule as records); feedback
  events in the log.
- **Questions:** RQ12, RQ14 (retrieval policy and frequency over long horizons).
- **Experiments:** adaptive vs fixed policies on identical streams; feedback ablations.
- **Artifacts:** policy-state snapshots; adaptation events.
- **Validation:** every adaptation is a replayable event; the policy state at any time is
  reconstructible from the log; runs reproduce from manifests.
- **Dependencies:** 6, 9.
- **Deferred:** trained neural rerankers beyond small pinned local models.

## Phase 14 — Memory contamination and adversarial experiments (L12)

- **Objective:** measure resilience to synthetic, controlled attacks.
- **Capabilities:** false-memory injection, delayed and repeated misinformation, source
  spoofing, conflicting-source injection, semantic and temporal poisoning, near-duplicate
  flooding, targeted interference, memory overload — all as `Intervention`s.
- **Components:** attack interventions extending Phase 3's framework; attack-aware
  outcome attribution through `taxonomy` and provenance.
- **Questions:** RQ5, RQ6, RQ10, RQ11.
- **Experiments:** attack × policy × retrieval grids with paired comparisons against
  unattacked baselines.
- **Artifacts:** intervention records identifying every injected experience; attributed
  failure shifts.
- **Validation:** attacks are synthetic and fully recorded; every attack-caused failure
  traces to an injected experience.
- **Dependencies:** 3, 11, 12.
- **Deferred:** attacks on real users or real data (out of scope permanently).

## Phase 15 — Long-horizon benchmark generation (L13)

- **Objective:** generated, content-addressed benchmarks with controlled difficulty.
- **Capabilities:** horizon length, memory density, contradiction rate, source
  reliability, semantic similarity, correction frequency, forgetting pressure,
  contamination rate and retrieval difficulty as generator parameters, with epistemic
  ground truth.
- **Components:** benchmark generators producing `Dataset`s; difficulty measurements.
- **Questions:** enables RQ2–RQ14 at scale.
- **Experiments:** per-parameter manipulation checks (does each parameter move its
  target property?).
- **Artifacts:** benchmark datasets and their generator specs.
- **Validation:** datasets reproducible from parameters; each parameter's effect measured.
- **Dependencies:** 9–14.
- **Deferred:** natural-language rendering beyond templates.

## Phase 16 — Large-scale longitudinal evaluation (L15)

**Status: partly delivered by Super-Phase 5.** Delivered: retention, stale-answer, evidence
retention and revision-latency curves with intervals, paired comparisons and onset detection on
five generated worlds (`retention`, `adaptive_study`). Still open: the same curves on Phase 15
benchmarks, incremental state reconstruction (the lab recomputes decisions per probe),
contamination and interference curves, cluster-aware statistics beyond replicate-level pairing,
and capacity.

- **Objective:** measurements over time rather than at single points.
- **Capabilities:** retention, forgetting, interference and contamination curves; error
  recovery; correction persistence; temporal degradation; retrieval latency; capacity.
- **Components:** evaluation at many time points; incremental state reconstruction
  (replacing the whole-log scan); curve estimation with intervals; cluster-aware
  statistics for correlated probes.
- **Questions:** RQ2, RQ7, RQ11, RQ13.
- **Experiments:** longitudinal runs on Phase 15 benchmarks.
- **Artifacts:** curve records decomposable to per-probe observations.
- **Validation:** every curve point traces to its probes; statistical methods checked
  against reference values.
- **Dependencies:** 15.
- **Deferred:** cross-study meta-analysis (19).

## Phase 17 — Memory autopsy (L14, flagship)

**Status: partly delivered by Super-Phase 6** ([results](experiments/superphase6-autopsy-replay-benchmark.md)).
Delivered (`autopsy`, `autopsy_demo`): an evidence-linked autopsy of memory answers and beliefs
(found / missing / not applicable links, verified by rebuilding), deterministic replay of stored
runs at a cutoff, fixed-evidence counterfactuals, and substrate replay. Still open: autopsy of
Phase 2-7 hybrid retrieval traces and of every stored response of an experiment, links to
interventions and experiments, and the API and observatory (Phase 20).

- **Objective:** answer "why did MEMORIA produce this memory or answer?" as a stored,
  reproducible research object.
- **Capabilities:** response → evaluation → trace → ranking signals → versions → graph
  lineage → consolidation history → experiences → sources → temporal context →
  interventions → experiment.
- **Components:** `autopsy` builder producing an `Autopsy` record from the graph and
  evaluations.
- **Questions:** RQ15.
- **Experiments:** autopsy completeness across all stored responses of a study.
- **Artifacts:** `Autopsy` records.
- **Validation:** every link resolves to a stored, verified artifact; an autopsy is
  reproducible from its inputs; missing evidence is reported as missing.
- **Dependencies:** 8, 12, 16.
- **Deferred:** interactive rendering (20).

## Phase 18 — Experiment orchestration and parameter sweeps

- **Objective:** declarative sweeps over any manifest variable.
- **Capabilities:** sweep specifications, resumable execution, result tables.
- **Components:** `SweepSpec` (a content-addressed set of manifests); resumable runner.
- **Questions:** all RQs that require dose–response designs.
- **Experiments:** sweep reproduction; interrupted-and-resumed equivalence.
- **Artifacts:** sweep manifests, result tables linking to run and evaluation digests.
- **Validation:** re-running a sweep reproduces every artifact; resumption is equivalent
  to an uninterrupted run.
- **Dependencies:** manifest schema v2 (§4.7), 3, 4.
- **Deferred:** distributed execution.

## Phase 19 — Advanced statistical and research analysis

- **Objective:** defensible inference beyond paired proportions.
- **Capabilities:** cluster-robust and mixed models for correlated probes; multiplicity
  control across comparisons; effect-size summaries; pre-registered analysis specs.
- **Components:** analysis specifications as records; `statistics` extensions.
- **Questions:** all RQs (inferential quality).
- **Experiments:** method validation on simulated data with known effects.
- **Artifacts:** analysis records citing evaluation digests.
- **Validation:** methods reproduce published reference values; coverage checked by
  simulation.
- **Dependencies:** 16, 18.
- **Deferred:** Bayesian hierarchical models unless justified by a question.

## Phase 20 — Interactive research observatory and API (L15)

- **Objective:** inspect stored evidence interactively.
- **Capabilities:** FastAPI surface with no logic of its own; observatory UI rendering
  only stored artifacts; autopsy explorer.
- **Components:** `api`, `observatory`.
- **Questions:** supports RQ15 in practice.
- **Experiments:** rendered numbers match stored artifacts (automated).
- **Artifacts:** none new; views over existing ones.
- **Validation:** every rendered number links to its artifact digest.
- **Dependencies:** 17.
- **Deferred:** multi-user deployment.

## Phase 21 — Reproducibility and research packaging

- **Objective:** portable research bundles.
- **Capabilities:** export and import of complete studies (artifacts, manifests,
  environment records, model identities); verification tooling.
- **Components:** bundle format and verifier.
- **Questions:** meta: can others reproduce MEMORIA's answers?
- **Experiments:** bundle re-verification and re-execution on a fresh machine.
- **Artifacts:** bundles.
- **Validation:** a bundle verifies and reproduces on a clean environment.
- **Dependencies:** 18, 20.
- **Deferred:** hosting.

## Phase 22 — Final benchmark and scientific validation

- **Objective:** a documented comparative study of memory architectures on the generated
  benchmarks, with every claim traced to artifacts.
- **Capabilities:** full study across architectures, policies, retrieval strategies and
  interventions.
- **Components:** none new; composition of all layers.
- **Questions:** RQ1–RQ15.
- **Experiments:** the pre-registered study.
- **Artifacts:** a published bundle and report.
- **Validation:** every reported answer reproducible from the bundle; negative and null
  results reported.
- **Dependencies:** all prior phases.
- **Deferred:** nothing further in this roadmap.
