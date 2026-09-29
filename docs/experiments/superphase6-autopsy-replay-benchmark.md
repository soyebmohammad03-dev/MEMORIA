# Super-Phase 6: memory autopsy, temporal replay, counterfactuals and the multi-seed benchmark

Run with `python -m memoria.benchmark_run var/bench` (`--quick`: 2 + 2 replicates). All numbers
come from the full run: manifest `sha256:dfaf4c3b…0857`, result `sha256:42e8eb9f…1f59`, 30 test
+ 8 calibration memory replicates and 10 + 5 belief replicates per environment, 1,450 stored runs (1,200 memory, 250 belief; calibration-only runs are not stored), 24 min 15 s of
wall time in one process. Reproduction: the runs of memory replicates 0, 15, 29 and belief
replicates 0, 9 (170 runs) were re-run from the manifest alone into an independent store: 170/170
digests identical, and the analysis recomputed from the stored runs is identical to the stored
one. The autopsy demonstration digest is identical in both stores
(`sha256:cbdb9096…8625`). Existing schemas, manifests and digests are untouched; the one change to
existing code (moving the simulator's decision logic into `retention.Memory`) was checked against
the 120 stored Super-Phase 5 runs of a replicate: identical digests.

## Questions and hypotheses (fixed before the full run)

1. **Autopsy**: can every sampled answer or belief be explained as an evidence chain whose
   every reference resolves, and does the rebuilt autopsy equal the stored one? Expected: yes on
   generated worlds (by construction), and missing provenance is reported, not filled.
2. **Replay**: does replaying stored records at recorded decision times reproduce the live
   answers exactly, for every system and environment? Expected: yes; any miss means future
   information leaked or a policy is nondeterministic.
3. **Counterfactual**: with the evidence held fixed, how many decisions does one component change,
   and how much does that differ from re-simulating the alternative (whose feedback differs)?
   Expected: a large gap for ranking changes, a small one for retention changes.
4. **Benchmark**: do the Super-Phase 5 and 11-12 conclusions (adaptive ranking helps through
   provenance; forgetting trades staleness for missing answers; recency-only and conservative
   belief policies cost correctness; calibration is recoverable) hold on new seeds and more
   varied worlds?

## Method

**Environments** (five, shared by both tracks): stable, drifting, repetitive, noisy, adversarial.
*Memory track*: the Super-Phase 5 worlds with seeded parameter jitter (change interval x0.75-1.25,
query rate x0.8-1.2, stray-report rate x0.8-1.2) and near-collision entities in every world
(16 keys); seed families 3 (test) and 4 (calibration), new. Eight systems: A static (keep all,
latest wins, the baseline), B adaptive ranking, C age window 30 d, D age window 90 d, E importance
gate with latest ranking, F gate with adaptive ranking, G stale-aware gate, H static
(age + provenance) importance. *Belief track*: Phase 11-12 policies A evidence count (baseline),
B source-weighted, C recency-aware, D temporal-validity, H conservative on belief worlds
stable / temporal-change / duplicated-evidence / source-disagreement / adversarial; seed families
7 (test) and 8 (calibration). The two tracks share environment names, not worlds.

**Measures.** Memory: correct-answer rate (retrieval accuracy against hidden truth on audit
probes every 10 days after day 30), stale-answer rate, no-answer rate, evidence retention,
contested-correct rate (probes where known evidence disagrees concurrently), importance
calibration (AUROC for use and for truth; ECE raw and recalibrated on separate seeds),
provenance completeness (item -> report text -> claim span). Belief: correct rate (of all
probes), coverage, selective accuracy, accuracy on contradiction-heavy keys, ECE raw and
recalibrated (isotonic fitted on calibration seeds), belief-trace completeness (I76). Both:
sampled autopsy completeness and replay exactness (below).

**Autopsy and replay sampling.** In every memory run 6 answered audit probes are explained by
`memory_autopsy` (complete? verified by rebuilding? same answer as live?) and every 10th recorded
decision (audits and user queries) is replayed from the stored records; in every belief run 5
answered probes get a `belief_trace` and the ledger is reconstructed at two cutoffs.

**Statistics.** The unit is the replicate (independent seed). Paired difference against the
baseline, Student-t interval (not bounded to [0, 1]), exact sign test, Holm within track x
environment x metric, an underpowered flag; a pooled family pairs (environment, replicate)
units ("ALL"). With 10 belief replicates the smallest reachable Holm-adjusted p (four
comparisons) is 0.0078, so belief effects can be missed. "Holm p 0.0000" means below 5e-5.

## Results

### Memory track (30 replicates per environment; mean over replicates [t 95%])

| environment | system | correct | stale | no answer | evidence retention | contested correct |
|---|---|---|---|---|---|---|
| stable | A static | 0.768 [0.735, 0.802] | 0.000 | 0.000 | 1.000 | 0.498 [0.417, 0.579] |
| stable | F gate-adaptive | 0.940 [0.915, 0.966] | 0.000 | 0.000 | 0.967 | 0.896 [0.838, 0.954] |
| drifting | A static | 0.603 [0.589, 0.617] | 0.285 [0.267, 0.303] | 0.000 | 1.000 | 0.412 [0.376, 0.449] |
| drifting | B adaptive rank | 0.655 [0.639, 0.672] | 0.284 [0.265, 0.303] | 0.000 | 1.000 | 0.575 [0.543, 0.607] |
| drifting | C age-30 | 0.566 [0.555, 0.576] | 0.225 [0.200, 0.250] | 0.112 | 0.820 | 0.376 |
| drifting | G stale-aware | 0.632 [0.618, 0.646] | 0.277 [0.260, 0.295] | 0.000 | 0.818 | 0.563 |
| repetitive | A static | 0.610 [0.592, 0.628] | 0.228 | 0.000 | 1.000 | 0.493 [0.455, 0.532] |
| repetitive | B adaptive rank | 0.631 [0.613, 0.649] | 0.229 | 0.000 | 1.000 | 0.648 [0.605, 0.692] |
| noisy | A static | 0.445 [0.431, 0.459] | 0.276 [0.258, 0.295] | 0.004 | 1.000 | 0.312 [0.283, 0.340] |
| noisy | B adaptive rank | 0.500 [0.484, 0.516] | 0.288 [0.268, 0.307] | 0.004 | 1.000 | 0.431 [0.397, 0.464] |
| adversarial | A static | 0.547 [0.513, 0.580] | 0.104 | 0.017 | 1.000 | 0.360 [0.239, 0.480] |
| adversarial | F gate-adaptive | 0.722 [0.689, 0.755] | 0.056 [0.048, 0.064] | 0.038 | 0.936 | 0.817 [0.732, 0.902] |

Paired comparisons against A (per-replicate differences, Holm-adjusted; full tables in the run's
report):

| | correct | stale | evidence retention | contested correct |
|---|---|---|---|---|
| B adaptive rank, ALL | **+0.092** [+0.078, +0.105] (148/2) | -0.006, no detected difference | 0 | **+0.258** |
| F gate-adaptive, ALL | **+0.095** [+0.081, +0.109] | -0.007, n.d.d. | **-0.071** | **+0.259** |
| H static importance, ALL | **+0.083** [+0.069, +0.098] | -0.007, n.d.d. | 0 | see report |
| G stale-aware, ALL | **+0.038** [+0.030, +0.046] | **-0.003** [-0.004, -0.001] | **-0.220** | **+0.220** |
| C age-30, ALL | **-0.260** [-0.305, -0.216] (0/138) | **-0.051** [-0.059, -0.044] | **-0.500** | **-0.171** |
| D age-90, ALL | **-0.183** [-0.221, -0.146] | **-0.007** | **-0.321** | **-0.114** |
| drifting: B / F | **+0.052 / +0.052** | -0.001 / -0.000 (n.d.d.) | 0 / **-0.084** | **+0.163** |
| drifting: G | **+0.029** | **-0.008** [-0.013, -0.002] (p 0.049) | **-0.182** | **+0.150** |
| adversarial: F | **+0.175** [+0.147, +0.204] (30/0) | **-0.048** [-0.056, -0.041] | **-0.064** | **+0.457** |
| adversarial: B | **+0.157** | **-0.041** | 0 | **+0.457** |

Bold: Holm p < 0.05 and not underpowered. The conclusions of Super-Phase 5 replicate on new
seeds: adaptive ranking raises correctness through provenance (H, the static heuristic, gets
+0.083 of B's +0.092 pooled), the stale rate is not detectably moved except by forgetting, and
forgetting trades staleness for missing answers (C: 88% no answer in the stable world, 86% in
the adversarial one). New here, and large: on probes where known evidence disagrees, adaptive
ranking answers correctly far more often (+0.258 pooled, +0.457 adversarial). Rows marked
"underpowered" for B and H on evidence retention are exact zeros (they never archive), not
low power.

### Belief track (10 replicates per environment; means [t 95%])

| policy (drifting) | correct | coverage | selective acc. | ECE raw | ECE recalibrated |
|---|---|---|---|---|---|
| A evidence count | 0.714 [0.663, 0.765] | 0.872 | 0.819 | 0.214 | 0.062 |
| B source-weighted | 0.775 [0.731, 0.819] | 0.958 | 0.809 | 0.171 | 0.077 |
| C recency-aware | 0.444 [0.384, 0.504] | 0.531 | 0.839 | 0.312 | 0.090 |
| D temporal-validity | 0.753 [0.711, 0.794] | 0.853 | 0.883 | 0.285 | 0.047 |
| H conservative | 0.472 [0.414, 0.530] | 0.492 | 0.961 | 0.233 | 0.064 |

Pooled over environments against A (units = 50 environment-replicates): B correct **+0.059**
[+0.044, +0.073] (41/0), ECE raw **-0.023** (better); C **-0.140** (1/47) with ECE **+0.078**
(worse); H **-0.227** in correct but **+0.210** selective accuracy (it answers half as often and
is right 95-99% of the time); D selective accuracy **+0.031** and correct +0.002 (n.d.d.), ECE raw
**+0.039** (worse). The Phase 11-12 finding that recency-aware revision is worst and that a
conservative policy buys accuracy with coverage holds on new seeds. Raw ECE of every policy
is 0.17-0.36 and recalibration brings it to 0.01-0.15, above the 0.007-0.031 pooled in Phase 12 (fewer calibration replicates here; source-weighted in the repetitive world: 0.149).

### Importance calibration (memory track)

AUROC for future use 0.83-0.99 (higher when the loop is active, B, than passive, A: 0.833 vs
0.907 drifting), AUROC for truth 0.65-0.81, raw ECE 0.28-0.48, recalibrated on separate seeds
0.002-0.021. Importance predicts use far better than truth in the stable world (0.962 vs 0.653)
and comparably elsewhere (0.845 vs 0.791 repetitive), as in Super-Phase 5.

### Autopsy completeness and replay exactness (H1, H2)

| environment | memory lineage | autopsy complete | autopsy verified | memory replay exact | belief trace complete | belief replay exact |
|---|---|---|---|---|---|---|
| stable | 104704/104704 | 1440/1440 | 1440/1440 | 34176/34176 | 250/250 | 900/900 |
| drifting | 128725/128725 | 1440/1440 | 1440/1440 | 34904/34904 | 250/250 | 900/900 |
| repetitive | 130209/130209 | 1440/1440 | 1440/1440 | 47288/47288 | 250/250 | 900/900 |
| noisy | 125017/125017 | 1440/1440 | 1440/1440 | 34056/34056 | 250/250 | 900/900 |
| adversarial | 103679/103679 | 1440/1440 | 1440/1440 | 30760/30760 | 250/250 | 900/900 |

7,200 sampled memory autopsies (all 7,200 also gave the answer the live run gave; checked on the stored runs), 181,184
replayed memory decisions, 1,250 belief traces and 4,500 belief reconstructions: no miss.
This is expected on generated worlds whose evidence is complete; it shows the chain and the
replay are consistent, not that they would be on incomplete real data. The failure path is
exercised by tests: an autopsy built without the originating report marks `experience` missing
and `complete=False`; a key with no retrievable item is explained as such; a tampered autopsy is
detected by rebuilding and by unresolved references.

### Autopsy example (drifting world, replicate 0, system F)

Answer `umbrella` for `benn.employer` at day 10.5 (an audit probe), complete, no missing link:
candidates #1 `item:r:benn.employer:0:2` (umbrella) and #2 `…:0:0` (globex); rank rule
argmax((1 - 0.5) freshness + 0.5 importance); signals: contradiction 0.393 (one
`contradiction_discovered` event), correction 1.000, frequency 0.000, provenance 0.950, recency
0.947, success 0.500, importance 0.649; retention `grace` under gate-0.5; claim
`claim:d92f8470…` by rule `employer_assert` (entity benn, span verified); originating report
`r:benn.employer:0:2`, "Benn works for Umbrella." (occurred 3.41, recorded 5.80); source
`clinic:2` (declared prior 0.95); not superseded by known evidence. After the cutoff, not seen by
the decision: umbrella again at 24.6, then vandelay at 47.6 and 56.6; later answers umbrella
(20.5) then the other umbrella item (30.5, 40.5). The belief autopsy for `ana.dose` under
source-weighted revision links answer -> decision (score 0.648, arithmetic ok) -> six candidate
beliefs -> per-evidence trust factors -> revision events -> claims, experiences, sources and
consolidated memories; `complete=True`.

### Temporal replay

Replay around a source correction (stale-aware system, `chen.home`, event at day 40.4954): the
item was `active / importance_above` when replayed at the recording time and
`archived / source_corrected` 0.001 day later; the two snapshots have different digests and
the live run's ledger digest and freeze were unchanged. Tests cover before/after a correction,
forgetting (age window), feedback, later contradictory evidence and stale-memory exposure (a
system whose importance outweighs freshness answers with a superseded item; latest-wins cannot),
substrate replay of a memory log and its consolidation (a hierarchy replayed at day 10 equals one
built from a log holding only what was known then; the log digest is unchanged), and the
agreement of truncated and bulk replay.

### Counterfactual autopsy (evidence held fixed; drifting world, replicate 0)

| change | decisions | changed | why | audit gained / lost | audit answers changed by re-simulation | fixed vs re-simulated differ |
|---|---|---|---|---|---|---|
| F: rank weight 0.5 -> 1.0 | 1600 | 767 (0.479) | ranking 767 | 8 / 180 | 516 | 480 |
| B -> F: keep all -> gate 0.5 | 1600 | 1 (0.001) | retention 1 | 0 / 0 | 1 | 1 |
| F: remove the correction component | 1600 | 59 (0.037) | importance 59 | 1 / 7 | 86 | 74 |

Each changed decision carries the two winners and their freshness and importance under both
configurations (or that the base winner is archived under the alternative). Reading: raising
the rank weight to 1 changes almost half of the recorded decisions and, judged against hidden
truth (analysis only), loses 180 correct audit answers while gaining 8; replacing keep-all with
the gate changes a single decision; removing the correction component changes 59. The
re-simulation column shows the loop: 480 of the audit answers that the fixed-evidence
recomputation gives differ from those of the re-simulated alternative for the rank-weight change
(H3 supported: a large gap for ranking, tiny for retention). **These are decision-level
recomputations on identical evidence, not causal estimates**: the alternative system would have
produced different uses and feedback, which the re-simulation column reflects. One replicate
and one world: the counts illustrate the method and are not a benchmark estimate.

## Failures, negative findings and limitations

- **Completeness by construction.** 100% autopsy completeness and replay exactness on generated
  worlds validates consistency, not robustness to incomplete provenance; that is shown only by
  designed tests.
- **Belief power.** 10 replicates: several belief comparisons are not significant after Holm
  (for example D against A in the adversarial world, p 0.07, correct -0.033) and the smallest
  reachable adjusted p is 0.0078.
- **The "contradiction-heavy" tag** covers every probe in some worlds (drifting, repetitive,
  adversarial), so the belief contradiction measure equals the correct rate there and is
  informative only in the stable and noisy worlds. The memory "contested" measure is defined from
  known evidence only and is a heuristic (concurrent values within 7 days).
- **Intervals** are Student-t over replicates and can leave [0, 1] (for example belief coverage
  0.975 [0.937, 1.013]); the sign test is exact and unaffected.
- **Pooling ("ALL")** averages heterogeneous environments; per-environment rows are the primary
  results.
- **Two tracks, two kinds of world**: memory correctness is on text-report worlds, belief
  correctness on structured-evidence worlds; they are not comparable and are not combined.
- **Stale-answer rate** is not moved by any adaptive system in drifting, repetitive or noisy
  worlds (n.d.d.), only by forgetting, at the cost of answers.
- **Autopsy scope**: memory answers (Super-Phase 5 systems) and Phase 11-12 beliefs. Phase 2-7
  retrieval traces are not autopsied here (their provenance walk is `graph.trace_provenance`).
- **Reproduction is representative** (170 of 1,450 runs re-run, including calibration-dependent
  belief runs), exact on those; the analysis is recomputed from all stored runs.
- **Cost**: 24 min for the full benchmark; belief runs dominate.
- Worlds, jitter ranges, declared weights and thresholds are generated or declared, not tuned or
  real; nothing here says what real users or real text would do.

## Deferred

Autopsy of hybrid-retrieval traces and of every stored response of a run, links to
interventions and experiments, and an API to browse autopsies (Phase 17/20); a larger, fully
parameterised benchmark generator and cluster-aware statistics (Phases 15, 19); belief-track
worlds with real text; full (not representative) reproduction as a scheduled job; adaptive
retrieval policies inside the hybrid pipeline (Phase 13).
