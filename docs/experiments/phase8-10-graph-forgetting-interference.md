# Phases 8–10 study: memory graph, forgetting and interference

> Can a provenance-preserving graph represent memory without collapsing evidence and
> inference (RQ-G1), and does its structure change retrieval, contradiction discovery and
> provenance (RQ-G2)? What does each forgetting policy do to retrieval, provenance,
> contradiction visibility and integrity (RQ-F1), and can a system forget operationally
> while keeping auditable history (RQ-F2)? How do memories interfere as the population
> grows (RQ-I1), which mechanisms are most damaging (RQ-I2), and can interference be
> measured apart from retrieval failure (RQ-I3)?

Every number below is read from stored artifacts. Rerun with

```bash
python -m memoria.memory_lab var/artifacts            # all eight worlds, ~2 h with reproduction
python -m memoria.memory_lab var/artifacts stable     # one world (interference study is always run)
```

The command runs the world matrix and re-runs it (`reproduce_lab`), stores and verifies
one graph per world, runs the interference study and re-runs it
(`reproduce_interference`), runs the interaction experiments and the demonstration, and
prints the report the tables below were taken from.

These are designed, generated worlds with a deterministic lexical-subword reference
embedder (hashed character n-grams). Effects are properties of this design and these
policies, not universal claims about memory systems.

## Identity and reproducibility

| Artifact | Digest |
|---|---|
| Lab spec (`phase8-10-memory-lab`) | `sha256:0dbce07701b8e84ba3a270404d576e2344037718d0546efc3ab0e603c6a14c42` |
| Lab study (world matrix) | `sha256:3878e51d68747d9e16279fef5c45dcde1d2c1144014a4d7494ff0b8ca4cee235` |
| Interference study (11,880 observations) | `sha256:2bf3f6f3df13ef50e976f4cf80664b12eebf5df92af30233c388ae0400eb61d3` |
| Demonstration (re-run twice in-process: same digest; the original matrix run predates the contradiction search moving to the raw-memory run) | `sha256:a7a87c4819accbf1980135db6271c6e3c8c392c884a0f55305fa8bf0b243dc97` |

The lab re-run (`reproduce_lab`) and the interference re-run (`reproduce_interference`)
were both classified identical. Full execution took 58 minutes on one laptop CPU.

## Design

**Worlds** (`memory_lab.WORLDS`, seeded; Phase 7 generator). Probes ask for one key at a
valid time, with `known_at` up to 30 days later; expectations follow the Phase 3 epistemic
standard and are never changed by interventions (I20).

| world | entities × attributes | horizon | change every | extra |
|---|---|---|---|---|
| stable | 3 × 3 | 360 d | 180 d | — |
| rapid | 3 × 3 | 360 d | 20 d | 5-day ingestion delays |
| contradictory | 3 × 3 | 360 d | 90 d | same-instant rivals 60%, wrong-then-corrected 30% |
| many-entities | 10 × 3 | 360 d | 90 d | includes near-collisions Ana/Anna, Ben/Benn; 2 probes per key |
| overlap | 5 × 3 | 360 d | 90 d | 4 restatements per period, 90% as free text |
| source-noise | 3 × 3 | 360 d | 90 d | rivals 30% × 2 forum copies, 40 irrelevant notes |
| long | 3 × 3 | 1080 d | 120 d | — |
| adversarial | 3 × 3 | 360 d | 60 d | 3 restatements, rivals 40% × 3 copies, negated notes, 30-day grid |

**Systems.** *base*: mixed retrieval (Phase 7: attribute, lexical and semantic signals with
equal weights, temporal filter, surface exposure). *consolidated*: + Phase 7 hierarchical
consolidation (temporal regime, entity/timeline/co-change abstraction, every 30 days).
*graph*: graph-hybrid retrieval (+ `graph` generator, `graph_claim`, `graph_entity`) over a
conservative graph rebuilt at every probe. *forgetting*: + validity forgetting (claims
whose timeline period has ended are forgotten; derived memories cascade). *interference*:
+ injected distractors (`inject`, 4 per key: a near-collision note "Anai's home is X", a
same-entity other-attribute fact, a same-key rival at another time, an other-entity
same-attribute fact; source class forum). *combined*: all four. A ladder adds one
variable at a time under interference: interference → +consolidation → +graph →
+forgetting (= combined).

**Ablations** (adversarial, contradictory, overlap; resolution also many-entities): graph
retrieval variants (B graph-assisted: graph generator only; C graph-only; D graph = hybrid;
without `graph_claim`; without `graph_entity`; with `graph_contradiction`); aggressive
resolution; no inferred value mentions; every forgetting rule; consolidation (temporal,
abstractive) under the graph; temporal filter off; provenance semantics (cascade vs retain;
active vs historical graph view); each interference kind alone.

**Statistics.** Single-variable comparisons: `evaluation.compare` (paired Newcombe
interval — the risk difference is the effect size — exact McNemar, Holm within world ×
family, `u` = underpowered). Retention (expected memory retrievable) with the same method.
Because probes about one key share memory, a key-level analysis averages each key's
probes and compares keys: mean difference, Student-t interval, exact sign test and d_z
(keys are the independent units; n = 9–30). Comparisons that change several variables are
*composite* and are not attributed to any one of them. Graph metrics pool nodes of one
graph and are descriptive. The interference study's units are independent replicates.

## Results: the world matrix

### Correct answers by system (known.correct, scorable known-value probes)

| world | base | consolidated | graph | forgetting | interference | combined |
|---|---|---|---|---|---|---|
| stable | 23/36 | **36/36** | 23/36 | **35/36** | 18/36 | 26/36 |
| rapid | 10/36 | **25/36** | 10/36 | **23/36** | 7/36 | **24/36** |
| contradictory | 11/27 | **22/27** | 11/27 | **19/27** | 5/27 | 19/27 |
| many-entities | 34/60 | **58/60** | 34/60 | **56/60** | **26/60** ↓ | **53/60** |
| overlap | 40/60 | **60/60** | 40/60 | **53/60** | **27/60** ↓ | 42/60 |
| source-noise | 14/35 | **28/35** | 14/35 | **30/35** | 10/35 | **28/35** |
| long | 13/36 | **33/36** | 13/36 | **30/36** | 12/36 | **29/36** |
| adversarial | 11/30 | 17/30 | 11/30 | 18/30 | 11/30 | 19/30 |

Bold = significant against base after Holm (combined: composite comparison, same method);
↓ = a significant decrease. Selected effects (risk difference [95% CI], Holm p; key level:
mean, n keys, sign-test p):

- **The graph changed no answer in any world** (graph vs base: +0.00 in all eight; every
  probe's answer identical). Removing `graph_claim` or `graph_entity`, aggressive
  resolution, dropping inferred value mentions, and the historical instead of the active
  view also changed nothing. Graph-only retrieval was never better (contradictory −0.04,
  overlap −0.05, adversarial −0.13 [−0.33, +0.08], all p = 1). **Mechanism:** the
  graph's signals are true for every memory about the query key — including the stale and
  rival values that are the actual source of error — so they add the same amount to the
  competitors that matter, and the attribute signal already carried that information.
- **Penalising contradiction proximity helped where rivals are dense**: `graph_contradiction`
  +0.30 [+0.10, +0.45] (p = 0.023) in contradictory, +0.17 (p = 0.19) in adversarial, 0 in
  overlap. It works by demoting memories whose claims have contradiction edges — which
  includes the contested truth when a same-instant rival exists — so it trades
  contradiction visibility in the ranking for correctness under this probe design.
- **Temporal consolidation reproduced its Phase 7 effect in new worlds** (+0.33 to +0.56 in
  seven worlds, e.g. long +0.56 [+0.36, +0.70], p = 7.6e-06; key level +0.56, n = 9, sign
  p = 0.0039, d_z = 2.29), not in adversarial (+0.20, p = 0.37). Its cost is integrity:
  answers from stale derived memories (integrity 19/36 in rapid, 29/36 contradictory,
  53/60 overlap).
- **Under interference the graph helped only after consolidation, and only in one world**:
  ladder +consolidation → +graph was +0.23 [+0.06, +0.38] (p = 0.047) in adversarial and
  exactly 0 in the other seven worlds.
- **Validity forgetting raised correctness and lost history.** Base → forgetting:
  +0.30 to +0.47 in six worlds (e.g. source-noise +0.46 [+0.24, +0.62], p = 5.8e-04; long
  +0.47, p = 4.6e-05; many-entities +0.37, p = 8.9e-06). Retention fell with it: rapid
  −0.47 [−0.63, −0.29] (p = 6.1e-05; 17 of 36 retrievable expected memories made
  unavailable), adversarial −0.33 (p = 0.0078; 10 of 30), overlap −0.20 (p = 0.002; 12 of
  60). **Mechanism:** the rule forgets values whose period has ended — exactly the
  answers to historical probes. Correct answers still rose because the remaining current
  values stop competing with superseded ones (the Phase 7 mechanism, reached by removal
  instead of bounded intervals).
- **Adding forgetting to consolidation + graph under interference lost history in every
  world**: retention −0.15 to −0.42 (Holm-significant in stable, rapid, many-entities,
  overlap, long, adversarial) while known.correct did not improve (−0.13 to +0.06, all
  p ≥ 0.45). The two mechanisms fix the same problem; stacked, only the cost adds up.
- **Interference** (injected distractors) lowered correctness in many-entities −0.13
  [−0.22, −0.04] (p = 0.016) and overlap −0.22 [−0.32, −0.10] (p = 7.3e-04); in the other
  worlds −0.03 to −0.22, not significant. The Phase 4 taxonomy attributes the failures to
  the injection (`contaminated`: 4–20 probes per world) — none to `wrong_memory`. Of the four
  injected kinds, only same-key rivals mattered (inject:rival −0.30, p = 3.1e-05 in
  overlap; −0.22, p = 0.12 in contradictory; 0 in adversarial); near-collision notes,
  other-attribute facts and other-entity facts changed nothing in any ablation world.
- **Composite effects are not the sum of their parts.** Base → combined was significant in
  rapid (+0.39), many-entities (+0.32), source-noise (+0.40) and long (+0.44), and
  not in stable (+0.08), contradictory (+0.30, p = 0.077), overlap (+0.03) or adversarial
  (+0.27, p = 0.11) — while losing retrievable expected memories in every world (5–24).

### Forgetting policies (ablation worlds; retention = expected memory still retrievable)

| rule | contradictory correct / Δretention | overlap correct / Δretention | adversarial correct / Δretention |
|---|---|---|---|
| none (base) | 11/27 | 40/60 | 11/30 |
| age (120 d) | 19/27 / 0 | **54/60** / −2 | 11/30 / 0 |
| fifo (60) | 18/27 / 0 | 52/60 / **−21** | 17/30 / −2 |
| recency suppression † | **24/27** / 0 | **60/60** / 0 | 22/30 / 0 |
| importance | 7/27 / −7 | **27/60** ↓ / **−22** | 7/30 / −4 |
| access | 16/27 / −6 | 50/60 / **−23** | 18/30 / −3 |
| validity | 19/27 / −2 | 53/60 / **−12** | 18/30 / **−10** |
| contradiction | 18/27 / −2 | 40/60 / 0 | 11/30 / 0 |
| contradiction + contested | **21/27** / −2 | 40/60 / 0 | 14/30 / 0 |
| provenance (keep 1) | 11/27 / 0 | 40/60 / 0 | 12/30 / 0 |
| hybrid (3 of 5 votes) | 16/27 / 0 | 45/60 / 0 | 11/30 / 0 |
| selective: entity ana | 12/27 / −6 | 37/60 / **−11** | 8/30 / −8 |
| selective: source forum | 14/27 / 0 | 40/60 / 0 | 14/30 / 0 |
| selective: derived only | 11/27 / 0 | 40/60 / 0 | 11/30 / 0 |

Bold = Holm-significant (known.correct or retention) against its baseline; † against the
same retrieval policy with the `suppression` signal and no forgetting. Δretention counts
probes whose expected memory was retrievable before and is not after (collateral
forgetting at probe level; for the entity rule, the target entity's probes are among them).

- **Soft suppression gave the largest gains and lost nothing**: +0.48 [+0.25, +0.65]
  (p = 0.0032) in contradictory, +0.33 (p = 2.5e-05) in overlap, +0.37 (p = 0.096) in
  adversarial, with zero memories made unavailable. Old reports stay retrievable and are
  outranked by recent ones — which is correct for these probes because most ask about the
  recent past (the Phase 7 caveat about recency applies: recency is not truth).
- **Importance-based forgetting was harmful** (overlap −0.22, p = 0.0027; retention −0.37,
  p = 5.7e-06; contradictory −0.15, adversarial −0.13), repeating the Phase 7 finding that
  the decomposed importance model does not track what probes need.
- **FIFO, access and validity trade history for current answers** (retention −0.20 to
  −0.38 in overlap, all Holm-significant).
- **Rules that protect evidence changed little**: provenance (keep the first report of each
  claim) and forum-source forgetting never lost a retrievable answer; derived-only
  forgetting changed nothing (L1 memories still answer).
- **Forgetting contested claims** (`forget_contested`) raised correctness in contradictory
  (+0.37, p = 0.023) by removing one side of same-instant disagreements: contradiction
  visibility falls to zero for those keys (tested); the gain is a loss of visible
  disagreement, not a resolution of it.

### Provenance semantics: cascade versus retain (with consolidation)

| world | cascade: correct / retention / integrity / hidden-evidence answers | retain: correct / retention / integrity / hidden-evidence answers |
|---|---|---|
| contradictory | 23/27 / 24/27 / 35/36 / 0 | 22/27 / 27/27 / 28/36 / 7 |
| overlap | 51/60 / 48/60 / 60/60 / 0 | 60/60 / 60/60 / 48/60 / 12 |
| adversarial | 13/30 / 20/30 / 34/36 / 0 | 17/30 / 30/30 / 27/36 / 7 |

Retaining derived memories whose evidence was forgotten restored retention (overlap +0.20,
p = 9.8e-04; adversarial +0.33, p = 0.0039) and, in overlap, correctness (+0.15,
p = 0.0078) — because those derived memories still carry the forgotten facts. They then
answered 7–12 probes from evidence the policy had forgotten, and answer integrity fell
(overlap 60/60 → 48/60). The record lists each such memory (`evidence_retained`), so the
influence is visible, not silent (RQ-F2). Choosing the graph's historical view instead of
the active view changed no answer: in these runs, graph signals read only the candidate's
own edges, and unavailable candidates are excluded before any signal is computed.

## Results: graph structure (RQ-G1)

Eight stored graphs (456–3,069 nodes), each rebuilt byte-identically from its sources.
Every memory and derived memory reaches an experience (unreachable evidence 0), every
observed claim is supported, and no temporal error diagnostic fired; the contradictory
world has 16 warnings (derived validity before evidence) and 16 info diagnostics
(retroactive corrections). Conservative resolution made 0 false merges; in many-entities
it kept Ana/Anna and Ben/Benn apart and left 8 similarity candidates unmerged. 49–508
memories per world are claim-unavailable (free text, inferred patterns) rather than
parsed. After validity forgetting, 63–95% of claims are unsupported in the active view.

## Results: interference (RQ-I1–I3)

24 independent replicates per arm, loads 0–128. At load 0 every target was retrieved, so
every later failure is attributable against the controlled baseline.

- Same-key mechanisms (proactive, temporal, contradiction, retrieval mix) degraded from
  load 1–4 under every policy; at load 128 target@1 was 2–6/24 (e.g. contradiction
  −0.83 [−0.93, −0.60], Holm p = 1.5e-05; RR degradation +0.82, d_z = 2.19). Top-5 score
  entropy approached ln 5: maximal competition among equally scored rivals.
- Frequency: repeating each contradicting report 4× lowered target@1 at load 1 from 13/24
  to 9/24.
- Retroactive interference was blocked by the temporal filter until load 64 (11/24), where
  the target was lost through candidate-generator truncation before filtering (15 "other"
  failures, no distractor at rank 1); without the filter, onset was load 1 (9/24).
- Entity interference affected only text-only retrieval (onset load 1, 1/24 at load 8).
- Semantic near-collisions and consolidation interference changed no retrieval up to load
  128 — but aggressive entity resolution falsely merged the target with a near-collision
  in 23 of 24 replicates (conservative: 0).
- Graph-hybrid retrieval behaved identically to mixed retrieval in every arm.

## Interactions (single designed worlds; observed effects, not universal causes)

- Forgetting forum sources (source-noise): cascade made 41 graph nodes unavailable and
  left 71 of 201 claims unsupported in the active view; retain made 20 unavailable, left
  10 unsupported, but kept 21 derived memories citing hidden evidence (7 dangling).
- Forgetting entity Ana (many-entities): precision and recall 1.0, 0 of 366 non-target
  memories affected, yet the active graph fragmented from 1 to 13 components.
- The graph linked 18 of 36 answers (contradictory) and 11 of 36 (adversarial) to a
  contradiction edge; for 11 of each, the graph showed contradicting memories that
  retrieval had not surfaced.
- Retaining derived memories after forgetting let them answer 7–12 probes from hidden
  evidence; integrity fell to 0.75–0.80.
- Consolidation under injected interference reduced contaminated answers in overlap
  (20 → 13) while producing 15 inferred co-change errors.

## Demonstration (contradictory world)

Experiences → episodic formation → hierarchical consolidation → graph → retrieval →
targeted forgetting of entity `ana` → graph rebuild → interference injection → retrieval →
provenance traversal. Four runs: raw-memory graph, graph with consolidation, + forgetting,
+ forgetting and injected distractors.

1. **Graph-discovered contradiction.** Probe `ana.employer-0`: the answering memory is
   linked to 2 contradiction edges; retrieval surfaced 9 counter-evidence memories, and the
   graph shows 1 contradicting memory retrieval did not.
2. **Forgetting changes retrieval.** 12 probes changed answer. Forgetting Ana's memories:
   precision 48/48, recall 48/48, 0 of 115 non-target memories affected, 84 graph nodes
   forgotten (none removed); the graph diff lists each.
3. **Measurable interference.** Adding distractors moved known.correct by −0.11 [−0.24,
   +0.03] (Holm p = 0.25, underpowered); probes answered by an injected or wrong-entity
   memory rose from 4 to 7.
4. **Provenance traversal.** From a retrieval to its answer memory, 17 nodes, 2 experiences,
   sources `clinic:contradictory-7` and `email:contradictory-8`, no broken links, no
   inferred links followed.
5. **Failures the intervention introduced.** 9 probes went from correct to wrong: Ana's dose
   probes (3 `unsupported`, 1 `stale`) and employer probes (`wrong_memory`), because
   forgetting the entity removed the memories that answered them and other memories filled
   the gap; injected rivals turned 3 `ben.dose` probes `contaminated`.
6. **Failures the intervention reduced.** `ana.home-0` and `ana.home-2` went from `stale`
   to `correct` when superseded reports became unavailable.

## Negative findings

The graph did not improve retrieval on its own; forgetting improved current answers by
destroying historical ones; stacking forgetting on consolidation added only cost;
importance-based forgetting harmed accuracy; a contradiction-proximity penalty "helps" by
hiding disagreement; near-collisions damage entity resolution, not retrieval.

## Known limitations

Generated worlds, template text and a lexical-subword embedder; surface-name heuristics;
co-mention is not negation-aware; probe-level intervals treat probes as independent (the
key-level analysis has only 9–30 clusters).
