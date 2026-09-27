# Phase 6 study: hybrid, explainable retrieval under disagreeing signals

> When multiple memory signals disagree, which memories are retrieved, why are they ranked
> that way, and how sensitive is the ranking to each signal?

This document records what was measured, where it came from, and what it does not show.
Every number is read from stored artifacts; rerun with

```bash
uv sync --extra neural --extra ann
python -c "from memoria.neural import MINILM, default_model_dir, fetch_model; fetch_model(MINILM, default_model_dir())"
python -m memoria.hybrid_eval var/artifacts minilm   # the study reported here
python -m memoria.hybrid_eval var/artifacts hashed   # control representation; no model needed
```

Each command runs the study, re-runs it (`reproduce_hybrid`), re-reads and re-validates
every stored trace (`verify_experiment`), and prints the report the tables below were
taken from.

## Identity

| Artifact | Digest |
|---|---|
| Benchmark (`hybrid-benchmark` v1) | `sha256:7aa1b23f696d24bd8c93bf4bb54857cc0fc7d741782e0bd3ec88a770412e571b` |
| Dataset (45 experiences) | `sha256:f560a8651cd63958801bfea9731cd92c9e7502dfb1fdc0ebe5ce5a02f49cc374` |
| Memory log export (episodic-v1) | `sha256:a47182101a5111ca56839be320e8c45e359ce2a5b9af8dceaddda32a347d5584` |
| Corpus identity (known at day 300) | `sha256:62873d5c1757a157bf8530b7046291002a96c4f6b5b20fa942e64ea12a29685b` |
| Study spec, MiniLM (41 policies) | `sha256:9464d5146bf8d230ddb1f701a21c91124eafd29f3fc913178d98d0c371ad1292` |
| Experiment, MiniLM | `sha256:39107a76e5ac4c9fccb42765e5b05c22b8379be469a87bd4b6a05b2bb5a04c4b` |
| Semantic index (MiniLM, exact) | `sha256:3b5fa64292e1a80137a92e8881603acad00bf07046f2fcc6a7a859dd92d89058` |
| Embedder (`onnx-sentence` v1, MiniLM-L6) | `sha256:ec4571c402e2159015075aae3051c2394e77b4dc58c67a7edce6b6961866236c` |
| Study spec, hashed control | `sha256:291ae52c46e89f77e420274a44a76ac8df8616a6b4e09d56cacd38c7fe9d1093` |
| Experiment, hashed control | `sha256:6e44d59d065112c2fca1ba43211f5869a25c411007d87d8bc18e28bd48ceaedd` |
| Reference policy `full` | `sha256:4066e2180afce32ed6d3c7a6ca3837fef81a54700b42ccff48e287fb1ca6fbe8` |
| Diversity-free hybrid `+contradiction` | `sha256:cc4c530c71d2c2d394bd3ef62e5df4eb71863680eeeaf8499abb91c4c73e2d21` |

**Reproducibility.** Two executions of the MiniLM study into separate, empty stores produced
the same experiment digest, and each in-process rerun was classified identical (macOS
arm64, Python 3.13, onnxruntime 1.30.0). The hashed-control study reproduced identically
too. Each store holds 1,353 `HybridTrace` artifacts (41 policies × 33 queries); every one
was re-hashed on load, re-validated (normalisation, contributions, relevance, penalties,
order, exposure, selection and explanations re-derived), and matched to its diagnostics;
`ArtifactStore.verify()` re-hashed every object in both stores. Timings differ between runs
and live in separate `PerformanceRecord`s, never in the experiment's identity.

## Design

**Benchmark** (`scenarios.hybrid_benchmark`). 45 experiences about seven subjects, ingested
when they occur and stored verbatim (episodic-v1). They mix statement-language facts
(`set ana.home = Berlin`, which carry structured claims) with free-text notes (no claim).
13 primary queries fix `known_at` = day 300, a `valid_at` and a claim key. Every judged
candidate has a role declared before any measurement: *target* (answers at `valid_at` by the
Phase 3 epistemic standard: latest report by occurrence holds; same-instant disagreement is
contested, so both values are targets; corrections are retroactive; a forget leaves no
target), *related* (same subject and attribute, not the answer), or *trap*. A test
recomputes every query's target values from the reports and checks them against the
declared roles. 20 variants perturb primary queries. Twelve adversarial cases are tagged on
candidates and queries.

**Policies** (`hybrid_eval.phase6_policies`, 41). Weights are always equal across a policy's
scored signals and sum to 1. That is the uninformed baseline, not a tuned setting.

- *Ablation ladder* (each step adds one component): `semantic` (its generator only),
  `lexical` (its generator only), `lexical+semantic`, `+temporal` (hard filter excluding
  expired, future, superseded and forgotten versions), `+recency` (exponential, half-life
  30 days on the record axis, inherited from Phase 2), `+source` (declared priors: clinic
  0.9, chat 0.7, email 0.7, forum 0.3; an unstructured source is *missing*), `+provenance`,
  `+attribute` (with the metadata generator), `+contradiction` (surface mode), `full`
  (+ MMR diversity, β = 1, i.e. λ = 0.5, embedding similarity).
- *Leave one out* against two references: `full`, and the diversity-free `+contradiction`
  (see the finding on diversity below).
- *Sensitivity*: contradiction modes (penalize, paired), reciprocal rank fusion (k = 60),
  temporal as a soft signal instead of a filter, recency half-life 7/90/365 days, diversity
  β ∈ {0.1, 0.25, 0.5, 2} and token-Jaccard similarity, lexical rank and z-score
  normalisation.

**Measurements** (primary queries; k = 5 unless stated). recall@k over targets; *fact
recall*@k over distinct target facts (the benchmark states each answer several times on
purpose, so plain recall rewards returning duplicates); precision@k; MRR over the full
ranking; nDCG@k with grades target 2, related 1, other 0; *exposure* = among queries whose
top result has a surviving memory asserting another value for its key, the fraction where
one is visible (selected or surfaced as counter-evidence); *redundancy* = selected results
repeating the fact of a higher-ranked result; *validity* = selected results valid at
`valid_at`. Proportions carry Wilson 95% intervals. Paired tests (Newcombe method 10, exact
McNemar, Holm within family × measure, underpowered flag) reuse `memoria.statistics`, on
(query, target) pairs for recall, (query, fact) pairs for fact recall, and queries for
success@1.

## Results (MiniLM representation)

### Ablation ladder

| policy | R@1 | R@3 | R@5 | fact R@5 | P@5 | MRR | nDCG@5 | exposure | redundancy | validity |
|---|---|---|---|---|---|---|---|---|---|---|
| semantic | 0.200 | 0.571 | 29/35 = 0.829 [0.673, 0.919] | 14/14 | 0.446 | 0.725 | 0.761 | 0/0 (undefined) | 24/65 = 0.369 | 52/65 = 0.800 [0.687, 0.879] |
| lexical | 0.171 | 0.514 | 26/35 = 0.743 [0.579, 0.858] | 14/14 | 0.406 | 0.690 | 0.689 | 1/1 | 17/64 = 0.266 | 55/64 = 0.859 [0.754, 0.924] |
| lexical+semantic | 0.200 | 0.600 | 28/35 = 0.800 [0.641, 0.900] | 14/14 | 0.431 | 0.771 | 0.761 | 1/1 | 21/65 = 0.323 | 54/65 = 0.831 [0.722, 0.903] |
| +temporal | 0.229 | 0.629 | 29/35 = 0.829 | 13/14 | 0.446 | 0.792 | 0.805 | 1/1 | 23/65 = 0.354 | 65/65 = 1.000 [0.944, 1.000] |
| +recency | 0.229 | 0.629 | 30/35 = 0.857 | 13/14 | 0.462 | 0.792 | 0.811 | 0/0 (undefined) | 24/65 = 0.369 | 65/65 |
| +source | 0.229 | 0.600 | 29/35 = 0.829 | 13/14 | 0.446 | 0.792 | 0.799 | 0/0 (undefined) | 22/65 = 0.338 | 65/65 |
| +provenance | 0.229 | 0.600 | 29/35 = 0.829 | 13/14 | 0.446 | 0.792 | 0.799 | 0/0 (undefined) | 22/65 = 0.338 | 65/65 |
| +attribute | 0.257 | 0.686 | 29/35 = 0.829 | 13/14 | 0.446 | 0.833 | 0.847 | 7/7 = 1.000 [0.646, 1.000] | 22/65 = 0.338 | 65/65 |
| +contradiction | 0.257 | 0.686 | 29/35 = 0.829 | 13/14 | 0.446 | 0.833 | 0.847 | 7/7 = 1.000 [0.646, 1.000] | 22/65 = 0.338 | 65/65 |
| full | 0.257 | 0.257 | 10/35 = 0.286 [0.163, 0.451] | 10/14 = 0.714 [0.454, 0.883] | 0.154 | 0.776 | 0.420 | 7/7 = 1.000 | 0/65 = 0.000 [0.000, 0.056] | 65/65 |

The last row is not "best": each row measures what one added component changed. Consecutive
paired tests on this benchmark:

- Only one step differs beyond chance after Holm adjustment. Adding diversity (`+contradiction` → `full`)
  lost 19 (query, target) pairs and gained none (recall@5 −0.543 [−0.679, −0.341],
  p = 3.8e-6, Holm p = 3.4e-5). In fact recall the same step is −0.214 [−0.474, +0.063]
  (3 facts lost; p = 0.25, underpowered).
- Every other step: |difference| ≤ 0.086 in recall@5 with intervals spanning zero. Most
  tests are flagged *underpowered*: with ≤ 5 discordant pairs the exact test cannot reach
  α = 0.05. The ladder's success@1 has 12 queries, so none of its tests can reach
  significance.
- The temporal filter changed validity from 54/65 to 65/65 selected results. It also
  removed the only target of `team-meeting-history` (see below), which is why fact recall
  drops from 14/14 to 13/14.

### Leave one out

| reference | removed | ΔR@5 | Δfact R@5 | ΔMRR | ΔnDCG@5 | Δexposure | Δredundancy | Δvalidity | affected queries | entered/left top-5 | mean displacement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | semantic | +0.029 | +0.071 | +0.141 | +0.040 | +0.000 | +0.000 | +0.000 | 3 | 6/6 | 0.200 |
| full | lexical | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 11 | 8/8 | 0.184 |
| full | recency | −0.057 | −0.143 | −0.143 | −0.055 | +0.000 | +0.000 | +0.000 | 4 | 6/6 | 0.167 |
| full | source | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 5 | 6/6 | 0.095 |
| full | provenance | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 3 | 2/2 | 0.057 |
| full | attribute | +0.000 | +0.000 | −0.040 | −0.049 | — | +0.015 | +0.000 | 13 | 26/26 | 0.500 |
| full | temporal | −0.057 | −0.143 | −0.132 | −0.119 | +0.000 | +0.000 | −0.385 | 12 | 31/31 | 0.374 |
| full | contradiction | +0.000 | +0.000 | +0.000 | +0.000 | −1.000 | +0.000 | +0.000 | 0 | 0/0 | 0.000 |
| full | diversity | +0.543 | +0.214 | +0.057 | +0.427 | +0.000 | +0.338 | +0.000 | 13 | 39/39 | 0.645 |
| +contradiction | semantic | −0.057 | +0.000 | +0.083 | −0.041 | +0.000 | −0.077 | +0.000 | 10 | 6/6 | 0.331 |
| +contradiction | lexical | +0.029 | +0.000 | +0.000 | +0.018 | +0.000 | +0.031 | +0.000 | 13 | 14/14 | 0.365 |
| +contradiction | recency | +0.000 | +0.000 | −0.083 | −0.017 | +0.000 | +0.000 | +0.000 | 4 | 1/1 | 0.237 |
| +contradiction | source | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 4 | 2/2 | 0.113 |
| +contradiction | provenance | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 | 0/0 | 0.000 |
| +contradiction | attribute | +0.000 | +0.000 | −0.042 | −0.048 | — | +0.000 | +0.000 | 13 | 1/1 | 0.994 |
| +contradiction | temporal | −0.114 | +0.071 | −0.069 | −0.144 | +0.000 | −0.092 | −0.385 | 12 | 25/25 | 0.439 |
| +contradiction | contradiction | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 | 0/0 | 0.000 |

"Which retrieval signals actually matter on this controlled benchmark?", as measured:

- **Diversity** had the largest measured effect of any component, in both directions:
  redundancy 0.338 → 0.000, and recall, fact recall and nDCG fell. It is the only paired
  difference that survives multiplicity adjustment.
- **Temporal filtering** changed the most memories (25–31 entered/left top-5 sets across 12
  queries) and moved temporal validity by 0.385. Its recall effects are within noise.
- **Attribute match** changed the order within the top 5 of all 13 queries (displacement
  0.99) while barely changing its membership (1 entered/left), and changed MRR by −0.04.
- **Contradiction handling (surface)** changes no ranking by construction. It moves only
  exposure (7/7 → 0/7 without it, in `full`).
- **Source and provenance** changed membership of a few top-5 sets (≤ 6) and no aggregate
  metric. Provenance is 1.0 for every structured source in this benchmark, so it can only
  separate the one unstructured import.
- **Recency** at a 30-day half-life gives every memory ≤ 0.006 (the corpus spans ~300 days
  before `known_at`). It acts only as a tie-breaker between otherwise equal memories, yet
  in `full` that tie-breaking changes what MMR selects (Δfact R@5 −0.143).
- **Removing the semantic signal** raised MRR (+0.141 with diversity, +0.083 without) on
  this benchmark. The success@1 differences behind it are 2 of 12 queries (p = 0.5).

### Contradiction handling, fusion, temporal and diversity variants

| policy | R@5 | fact R@5 | MRR | nDCG@5 | exposure | redundancy | validity |
|---|---|---|---|---|---|---|---|
| full (surface) | 0.286 | 10/14 | 0.776 | 0.420 | 7/7 | 0/65 | 65/65 |
| contradiction:penalize | 0.314 | 11/14 | 0.917 | 0.457 | 0/5 = 0.000 [0.000, 0.434] | 1/65 | 65/65 |
| contradiction:paired | 0.371 | 13/14 | 0.833 | 0.521 | 7/7 | 0/65 | 65/65 |
| fusion:rrf (k = 60) | 0.314 | 11/14 | 0.794 | 0.414 | 7/7 | 0/65 | 65/65 |
| temporal:soft (no filter) | 0.257 | 9/14 | 0.776 | 0.409 | 11/11 | 0/65 | 58/65 = 0.892 [0.794, 0.947] |
| recency:7d | 0.229 | 8/14 | 0.631 | 0.365 | 7/7 | 0/65 | 65/65 |
| recency:90d | 0.286 | 10/14 | 0.842 | 0.432 | 7/7 | 0/65 | 65/65 |
| recency:365d | 0.314 | 11/14 | 0.917 | 0.460 | 7/7 | 0/65 | 65/65 |
| diversity β = 0.1 | 0.714 | 13/14 | 0.833 | 0.764 | 7/7 | 16/65 = 0.246 | 65/65 |
| diversity β = 0.25 | 0.629 | 13/14 | 0.833 | 0.683 | 7/7 | 13/65 = 0.200 | 65/65 |
| diversity β = 0.5 | 0.314 | 10/14 | 0.804 | 0.458 | 7/7 | 2/65 = 0.031 | 65/65 |
| diversity β = 2 | 0.286 | 10/14 | 0.773 | 0.420 | 7/7 | 0/65 | 65/65 |
| diversity token-Jaccard (β = 1) | 0.371 | 10/14 | 0.802 | 0.489 | 7/7 | 3/65 | 65/65 |
| lexical rank-v1 | 0.286 | 10/14 | 0.776 | 0.420 | 7/7 | 0/65 | 65/65 |
| lexical zscore-v1 | 0.400 | 11/14 | 0.799 | 0.477 | 3/3 | 3/65 | 65/65 |

- **Penalize** demotes *both* sides of a same-instant contradiction: on `leo-dose` the
  clinic claim fell from rank 1 (`full`) to 9 and the forum claim from 13 to 14. It hides the
  disagreement (exposure 0/5), which is why surfacing, not penalising, is the default.
- **Paired** exposure places the other value next to its primary. On `ana-home-now` the
  superseded "Paris" claim ranks first and the current "Berlin" claim is placed second,
  although MMR had pushed it down (final −0.002 after a 0.743 redundancy penalty).
- **Diversity β** traces a trade-off rather than an optimum. As β goes 0.1 → 2, redundancy
  falls from 0.246 to 0 and fact recall from 13/14 to 10/14. At β ≥ 0.5 the redundancy
  penalty (MiniLM similarities between on-topic memories ≈ 0.6–0.8) outweighs the spread
  of relevance among on-topic candidates (≈ 0.2 under six equal weights). MMR then prefers
  off-topic memories, whose relevance is held up by query-independent signals (provenance
  and source contribute up to 0.32 to *every* candidate). The β = 1 used for `full` is the
  symmetric MMR convention, not a recommendation. This sweep is the evidence against
  treating it as neutral.
- **Recency half-life** changes rankings through MMR tie-breaking (MRR 0.631 at 7 days,
  0.917 at 365 days). That is sensitivity, not evidence that recency identifies correct
  memories.

### Temporal retrieval

- `ana-home-history` (valid day 30) and `priya-history` (day 100) are old-but-exact queries.
  With the temporal filter the old targets rank 1 (`+contradiction`: `chat:a0` rank 1,
  `email:p30` rank 1). Without it (`full-temporal`), later memories that are not yet valid
  outrank them (`chat:a0` rank 15, `email:p30` rank 3).
- `team-meeting-history` (valid day 25) exposes a **mismatch between memory-level validity
  and retroactive corrections**. The correction (`correct team.meeting = Thursday 4 pm`,
  occurred day 30) applies retroactively under the Phase 3/4 standard, so it is the target.
  But as an episodic memory it is valid only from day 30, so the temporal filter excludes it
  as *future*: 0 targets retrieved with the filter, rank 1 without it. The trace records
  the exclusion reason (`future`) for both correction memories.
- As a soft signal instead of a filter, temporal compatibility kept 7 non-valid memories in
  the selected results (validity 58/65) and lost more fact recall (9/14) than filtering did.

### Counterfactual retrieval (top-5 overlap with the base query; top-1 unchanged)

| policy | paraphrase | reorder | irrelevant wording | entity | attribute | time | negation | numeric |
|---|---|---|---|---|---|---|---|---|
| *invariant?* | yes | yes | yes | no | no | no | no | no |
| n | 3 | 4 | 3 | 3 | 1 | 2 | 2 | 2 |
| semantic | 1.000; 2/3 | 1.000; 4/4 | 1.000; 3/3 | 0.467; 1/3 | 1.000; 0/1 | 1.000; 2/2 | 1.000; 2/2 | 1.000; 2/2 |
| lexical | 0.733; 1/3 | 0.750; 3/4 | 0.800; 1/3 | 0.200; 1/3 | 0.200; 0/1 | 1.000; 2/2 | 1.000; 2/2 | 0.900; 2/2 |
| lexical+semantic | 0.600; 2/3 | 0.950; 4/4 | 0.867; 1/3 | 0.333; 1/3 | 0.800; 0/1 | 1.000; 2/2 | 1.000; 2/2 | 0.900; 2/2 |
| +contradiction | 0.733; 3/3 | 0.900; 4/4 | 0.867; 2/3 | 0.467; 0/3 | 0.600; 0/1 | 0.300; 1/2 | 1.000; 2/2 | 0.800; 1/2 |
| full | 0.533; 3/3 | 0.700; 4/4 | 0.667; 2/3 | 0.200; 0/3 | 0.400; 0/1 | 0.600; 1/2 | 0.900; 2/2 | 0.500; 1/2 |

- MiniLM-only retrieval was perfectly stable under the three invariant perturbations. It
  was equally stable where the information need changed: the same top-5 when the attribute
  changed (0/2 of the new targets found), the time changed, the query was negated, or a
  number changed. Stability and sensitivity pull in opposite directions, and cosine
  similarity alone does not separate them.
- Lexical retrieval was less stable under invariant perturbations (overlap 0.73–0.80) and
  more responsive to entity and attribute changes (0.20).
- Adding the query key and the temporal filter (`+contradiction`) made time changes
  effective (overlap 0.30, all 4 new targets found). The negated question was answered
  exactly as the plain one (overlap 1.000) by every policy: no Phase 6 signal represents
  negation.

### Adversarial cases (`+contradiction` and `full`; n = queries tagged)

| case | n | focus visible (+contradiction / full) | target above focus (+contradiction / full) |
|---|---|---|---|
| lexically near-identical, not yet valid | 3 | 0/6 / 0/6 (filtered as future) | 3/3 / 3/3 |
| semantically similar, contradictory | 2 | 4/4 / 4/4 | — |
| recent but irrelevant | 5 | 1/5 / 0/5 | 5/5 / 5/5 |
| old but exact | 2 | 4/4 / 2/4 | — |
| redundant cluster (redundancy in top 5) | 2 | 5/10 / 0/10 | — |
| wrong entity | 5 | 2/6 / 1/6 | 5/5 / 4/5 |
| wrong attribute | 3 | 3/4 / 0/4 | 2/2 / 2/2 |
| low-overlap paraphrase | 3 | 0/3 / 0/3 | — |
| superseded value | 4 | 7/11 / 4/11 | 2/4 / 2/4 |
| source conflict | 1 | 2/2 / 2/2 | — |
| retroactive correction | 1 | 0/2 / 0/2 | — |
| retracted (forgotten) | 1 | 2/2 / 1/2 | — (no target exists) |

*Focus visible* for a trap is exposure to the trap. For a target or related memory it is
retrieval of it. Two cases are clear failures that the traces explain. The low-overlap
paraphrases ("Home for Ana is the German capital these days.", "Peanuts make Noah's throat
swell up.") never reach the top 5, because each query has four or more other targets that
share its words. The retroactive correction is filtered as future. In the source conflict
(clinic 20 mg vs forum 40 mg at the same instant), adding the source signal moved the clinic
claim from rank 3 to 2 and the forum claim from 2 to 3, and both stayed visible. The source
prior ordered the conflicting claims; it did not decide which is true.

### Example explanation (trace `sha256:28885f1a…`, policy `+contradiction`, `ana-home-now`)

Generators: lexical 13, metadata 3, semantic 20 proposals; union 20; 6 excluded as
`future` (the Munich move, the bicycle note, and others dated after day 100).

| rank | memory | attribute | lexical | provenance | recency | semantic | source | relevance |
|---|---|---|---|---|---|---|---|---|
| 1 | `set ana.home = Paris` (day 0) | +0.1667 | +0.1667 | +0.1667 | +0.0002 | +0.1290 | +0.1167 | 0.746 |
| 2 | `set ana.home = Berlin` (day 55) | +0.1667 | +0.1667 | +0.1667 | +0.0006 | +0.1231 | +0.1167 | 0.740 |
| 3 | "Ana lives in Berlin now." (day 80) | omitted | +0.1544 | +0.1667 | +0.0010 | +0.1469 | +0.1167 | 0.586 |

Rank 1's recorded explanation: *"relevance 0.746, final 0.746: largest contributions:
attribute +0.167, lexical +0.167; valid at the query's valid time; a later claim changed the
value; proposed by lexical, metadata, semantic; surfaced 1 memories with another value for
its key."* The superseded Paris claim outranks the current Berlin claim by 0.006, entirely
through the semantic contribution. In surface mode that is *shown*, not hidden: the reason
codes carry `conflict:superseded`, and the Berlin claim is attached as counter-evidence.
Relevance here is a retrieval-policy score, not a judgement of which value is true.

### Statistics, stated plainly

On this benchmark (13 primary queries; 35 (query, target) pairs; 14 (query, fact) pairs;
12 queries with a target), only the diversity step (ladder and leave-one-out) differs from
its comparison beyond chance after Holm adjustment. Most other tests are underpowered by
construction: few discordant pairs, or at most 12 queries for success@1. Pairs share
queries, so intervals may be optimistic (ARCHITECTURE.md §4.5).

### Performance (median per query, macOS arm64, one thread; environment-bound)

| policy | generate | filter | features | score | rerank | explain (incl. trace validation) | total | candidates | traced Python peak |
|---|---|---|---|---|---|---|---|---|---|
| lexical | 1.4 ms | 0.6 ms | 0.2 ms | 0.08 ms | 0.6 ms | 0.5 ms | 3.3 ms | 12 | 233 KiB |
| semantic | 10.7 ms | 0.9 ms | 6.3 ms | 0.14 ms | 1.0 ms | 0.9 ms | 20.0 ms | 20 | 274 KiB |
| lexical+semantic | 11.5 ms | 1.1 ms | 0.6 ms | 0.25 ms | 1.1 ms | 1.1 ms | 16.1 ms | 23 | 483 KiB |
| full without diversity | 12.2 ms | 1.1 ms | 0.9 ms | 0.56 ms | 1.0 ms | 1.5 ms | 17.1 ms | 23 | 836 KiB |
| full | 11.5 ms | 1.1 ms | 0.8 ms | 0.55 ms | 3.7 ms | 1.5 ms | 19.7 ms | 23 | 843 KiB |

The semantic generator's cost is embedding the query (one MiniLM pass ≈ 10 ms). The feature
stage embeds the query again through the engine's own cache. The first policy to run
(`semantic`) pays for that; later policies hit the cache. Greedy MMR adds ~2.7 ms at 23
candidates (quadratic in candidates). Nothing was optimised.

### Hashed-embedding control (same benchmark, `hashed-char-ngrams` instead of MiniLM)

| policy | R@5 | fact R@5 | MRR | nDCG@5 | redundancy |
|---|---|---|---|---|---|
| semantic (hashed) | 0.600 | 12/14 | 0.684 | 0.628 | 0.308 |
| lexical+semantic | 0.829 | 14/14 | 0.778 | 0.761 | 0.354 |
| +contradiction | 0.829 | 13/14 | 0.875 | 0.848 | 0.338 |
| full | 0.371 | 10/14 | 0.843 | 0.509 | 0.062 |

Under the hashed representation, similarities between on-topic memories are lower, so MMR
at β = 1 removed less (redundancy 0.062 against 0.000 with MiniLM). The diversity trade-off
depends on the representation's similarity scale, not only on β.

## What this shows, and what it does not

- Under benchmark `hybrid-benchmark` v1 and the evaluation above, the diversity stage
  produced the largest measured change: less redundancy, lower (fact) recall. The temporal
  filter produced the largest change in temporal validity (0.831 → 1.000 of selected results).
  Most other single-signal changes are within noise at this sample size.
- It does **not** show that hybrid retrieval is better, that any weighting is right, or that
  any signal is useless in general. The benchmark is a controlled probe of 13 queries with
  roles designed by the author, not a sample of real memory use.
- Retrieval relevance is not truth. The top result for "Where does Ana live?" at day 100
  was a superseded claim under every policy that surfaces rather than penalises conflict.
  The engine's job in Phase 6 is to show that, and why; resolving it is Phase 11.

## Findings that changed the design during the phase

1. *Benchmark timing.* The recomputed-ground-truth test found that two free-text
   restatements in the contested subjects were dated a day after the conflicting claims.
   Under "latest report holds" they would have resolved the contest, so they were re-dated
   to the same instant before any reported run.
2. *Fact recall.* Plain recall rewards returning copies of one fact, which penalises
   diversity by construction, so fact-level recall was added.
3. *Two leave-one-out references.* Against `full`, diversity dominated every other
   removal, so the diversity-free `+contradiction` is a second reference.
4. *Counter-evidence wording.* Explanations first called older values "disputing". The
   relation is "asserts another value for the key", and the text now says exactly that.
