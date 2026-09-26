# Phase 5 study: lexical-subword versus neural representation

This records what was measured, where it came from, and what it does not show. Every
number below is read from stored artifacts; rerun with

```bash
uv sync --extra neural --extra ann
python -c "from memoria.neural import MINILM, default_model_dir, fetch_model; fetch_model(MINILM, default_model_dir())"
python -m memoria.semantic_eval var/artifacts
```

## Identity

| Artifact | Digest |
|---|---|
| Study spec (`phase5-lexical-vs-neural`) | `sha256:c1ab2a94244c8dd0d28ca382f9128aab502454a5c3c9aab73379cbb83a627fd7` |
| Experiment record | `sha256:caf0fac4ac9447ceac8efd75e855ce774f54c7f094e8386aed174bdd562df2bc` |
| Memory state indexed | `sha256:cdaec46f8cda03361cdd6516e96e7863bbc52e78e7d6a641e34900887308e5e4` |
| Reference embedder (`hashed-char-ngrams` v1, 256-d) | `sha256:31db1748…` |
| Neural embedder (`onnx-sentence` v1, 384-d) | `sha256:ec4571c4…` |

The neural spec pins `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0), `onnx/model.onnx`
(SHA-256 `6fd5d72f…`, 90,405,214 bytes) and `tokenizer.json` (`be50c362…`); mean pooling,
256 tokens, right truncation, L2 normalisation, ONNX Runtime CPU, one text per batch.

Two independent executions into separate stores produced the same experiment digest;
`reproduce_semantic` classified each rerun as **identical** (macOS arm64, Python 3.13,
onnxruntime 1.30.0).

## Design

51 candidate memories (verbatim episodic memories of 51 experiences) about seven subjects;
14 queries, two per subject — one worded like the facts, one reworded. Each candidate's
relationship to its subject's queries was labelled before any measurement (see
`memoria.scenarios.semantic_diagnostic`). *Relevant* = about the query's subject and
attribute, including contradictions and old values; entity substitutions and distractors
are not relevant. Retrieval policy: top-k by exact cosine (the simplest policy; hybrid
ranking is Phase 6).

## Representation: mean similarity to the query, by designed relationship

| Relationship | n | hashed | MiniLM |
|---|---|---|---|
| paraphrase | 28 | 0.217 | 0.739 |
| equivalent (little shared wording) | 12 | 0.165 | 0.651 |
| same fact, other source | 4 | 0.266 | 0.777 |
| temporal variant | 8 | 0.203 | 0.708 |
| contradiction | 8 | 0.191 | 0.715 |
| negation | 8 | 0.253 | 0.610 |
| numeric change | 6 | 0.236 | **0.761** |
| entity substitution | 10 | 0.089 | 0.468 |
| lexical distractor | 8 | 0.114 | 0.492 |
| shared vocabulary | 10 | 0.064 | 0.229 |
| unrelated | 612 | 0.021 | 0.110 |

Queries whose least-similar meaning-preserving variant beats every meaning-changing one
(negation, number, entity, contradiction, temporal): hashed 1/14, MiniLM 3/14
(Wilson 95% [0.08, 0.48]).

## Retrieval (top-k by cosine)

| | hashed | MiniLM |
|---|---|---|
| MRR | 0.929 | 1.000 |
| recall@1 (attainable 14/74) | 13/74 | 14/74 |
| recall@3 (attainable 42/74) | 31/74 = 0.419 [0.313, 0.533] | 42/74 = 0.568 [0.454, 0.674] |
| recall@5 (attainable 66/74) | 40/74 = 0.541 [0.428, 0.649] | 64/74 = 0.865 [0.769, 0.925] |
| precision@5 | 40/70 = 0.571 | 64/70 = 0.914 |
| false matches in top 5 | 24 unrelated, 3 entity, 2 lexical, 1 vocabulary | 4 entity, 2 lexical |
| misses at 5 | 14 paraphrase, 5 equivalent, 5 temporal, 3 negation, 3 number, 3 contradiction, 1 source | 4 negation, 3 equivalent, 2 paraphrase, 1 temporal |

Paired over the 74 (query, relevant memory) pairs, MiniLM minus hashed recall:
k=1 +0.014 [−0.112, 0.138], exact McNemar p = 1; k=3 +0.149 [0.006, 0.282], p = 0.061;
k=5 +0.324 [0.185, 0.448], p = 1.9e-5. At k=3 the Newcombe interval excludes zero while the
exact test does not reach 0.05; both are reported. Pairs share queries, so intervals may be
optimistic. Mean top-5 overlap between the representations: 0.56.

## What this shows — and does not

- The two representations are measurably different regimes: the reference is dominated by
  surface overlap (its false matches are mostly unrelated texts; it misses rewordings),
  the neural encoder by topical similarity.
- **Topical similarity is not truth.** MiniLM places numeric changes (0.761),
  contradictions (0.715) and temporal variants (0.708) as close as true paraphrases
  (0.739), and negations only slightly further (0.610). Retrieving the right subject does
  not identify the right value; that is the job of later stages (Phases 6, 11, 12).
- A diagnostic of 14 queries and 51 memories is a controlled probe, not a benchmark.
  Nothing here says which representation is better for a memory system in general.

## Tolerance evidence

The adapter was compared with the reference `sentence-transformers` implementation
(PyTorch, same revision) on five texts including Unicode, the empty string and a 400-word
input truncated at 256 tokens: maximum absolute difference per component 3.5e-7, cosine
≥ 0.9999999. Repeated runs of the adapter were byte-identical. The declared tolerance
(max_abs 1e-4, min_cosine 0.9999) is ~300× the cross-runtime difference; the implied
similarity bound (2·√384·1e-4 ≈ 0.004) is smaller than typical gaps between neighbouring
similarities in this study. Cross-*hardware* equivalence (e.g. x86-64 versus arm64) has not
been measured yet: CI does not download the model.

## Exact versus approximate (FAISS HNSW), 5,000 MiniLM vectors, 60 queries, k = 10

| efSearch | recall@10 | top-1 agreement | mean rank displacement |
|---|---|---|---|
| 4 | 0.852 | 0.917 | 0.25 |
| 16 (FAISS default) | 0.983 | 1.000 | 0.06 |
| 64 | 1.000 | 1.000 | 0.00 |

Approximate search only proposes candidates; every returned score is the exact cosine.

## Scaling (macOS arm64, one thread; environment-bound)

Measured with `memoria.semantic_eval.profile_scaling` over `synthetic_state(n)`.

| Memories | embed/text hashed | embed/text MiniLM | exact query MiniLM | HNSW query MiniLM | vectors MiniLM |
|---|---|---|---|---|---|
| 100 | 0.89 ms | 9.8 ms | 13 ms | 7 ms | 150 KB |
| 1,000 | 0.90 ms | 10.2 ms | 77 ms | 15 ms | 1.5 MB |
| 5,000 | 0.93 ms | 10.5 ms | 359 ms | 47 ms | 7.5 MB |

Exact search is the unoptimised pure-Python reference; its cost is linear in the number
of memories. Peak process memory with the model loaded: ~606 MB.

## A dependency finding

ONNX Runtime 1.30's POSIX build includes an HTTP telemetry client. During validation,
about one in four full test runs aborted at interpreter exit; the native crash backtrace
showed the runtime's telemetry worker thread handling an HTTP upload response while the
runtime was tearing down. MEMORIA now sets `ORT_DISABLE_TELEMETRY=1` before onnxruntime
initialises (the runtime's `disable_telemetry_events()` alone did not stop it): 0 aborts
in 20 runs afterwards, against 5 in 20 before. A snapshot of open sockets during
inference was inconclusive either way (uploads are short-lived); the evidence is the
backtrace and the change in abort rate.
