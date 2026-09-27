# Phase 7 study: consolidation, abstraction and the stability–plasticity trade-off

> How does memory consolidation strategy affect long-horizon retrieval quality, provenance
> fidelity, information loss, contradiction handling, computational cost and downstream
> memory reliability?

Every number below is read from stored artifacts. Rerun with

```bash
python -m memoria.consolidation_eval var/artifacts            # all eight worlds, ~40 min
python -m memoria.consolidation_eval var/artifacts baseline   # one world
```

The command runs the lab, re-runs it (`reproduce_lab`), runs the end-to-end
demonstration, and prints the report the tables were taken from.

## Identity and reproducibility

| Artifact | Digest |
|---|---|
| Lab spec (`phase7-consolidation-lab`) | `sha256:fcc9fa21833a4b5bc7a56f06ba29c033179bc60233fa2cec1515ce9c58f29ec7` |
| Lab study (170 runs, 170 evaluations, 162 paired comparisons) | `sha256:c6b7ee668590f919145a673cd2f474859eee5e7e964001d6f020e80960f5e809` |
| Demonstration | `sha256:880d58872ce981f15fd5ff77db5f3356843c42800892043d543870236f0c0dc7` |
| Representation | `hashed-char-ngrams` v1, 256-d (`sha256:31db1748…`), exact scan |

Two executions into separate, empty stores produced the same spec, study and
demonstration digests. Each in-process rerun was classified identical, and each hierarchy
of each run was re-consolidated from the log byte for byte (`replayed = True` for every
run). `ArtifactStore.verify()` re-hashed every object in both stores. All components are
deterministic: the lab uses the reference embedder, so "identical" means byte-identical.
A neural embedder would fall under I37's equivalence classes; that was not run here.
Timings live in separate `PerformanceRecord`s and differ between runs.

## Design

**Worlds** (`scenarios.PHASE7_WORLDS`, seeded). Three entities × three attributes (home,
employer, dose) = 9 keys, 36 probes each (4 per key; `known_at` up to 30 days after
`valid_at`; expectation by the Phase 3 epistemic standard).

| world | horizon | change every | extra |
|---|---|---|---|
| baseline | 360 d | 90 d | 2 restatements per period |
| short / long | 90 / 1080 d | 45 / 120 d | — |
| contradiction | 360 d | 90 d | same-instant rival reports 50%, wrong-then-corrected 30% |
| drift | 360 d | 25 d | 15-day ingestion delays, temporary states 30% |
| repetitive | 360 d | 90 d | 6 restatements per period |
| noisy | 360 d | 90 d | 60 irrelevant notes, 10-day delays |
| adversarial | 360 d | 60 d | 3 unreliable rival copies (poisoned repetition) 40%, negated notes, changes snapped to a 30-day grid (coincidental co-changes), one designed coupling |

Every report carries a ground-truth label (key, value, true period) that consolidation
never sees. Two reports are equivalent iff they share all three.

**Conditions.** A: no consolidation (the baseline of every comparison). B: lossless
(exact text). C: claim dedup, and canonical dedup. D: abstractive (semantic ≥ 0.5 plus
entity profiles). E: hierarchical (temporal plus entity, timeline and co-change). F:
recency-driven (claim; only memories recorded in the last 90 days are promoted). G:
importance-driven (claim; six equally weighted components, threshold 0.5). H: hybrid
(temporal, importance and all abstractions). Also a no-op policy. All use mixed retrieval
(levels L1–L4; attribute, lexical and semantic signals; temporal filter; surface
exposure) and consolidate every 30 days. Ablations run on the adversarial, baseline and
contradiction worlds; stability–plasticity sweeps on baseline, drift and long; retrieval
modes on baseline, drift, long and repetitive.

**Statistics.** known.correct is compared by `evaluation.compare`, retention
(known.expected_in_state) with the same Phase 4 functions. Both use paired Newcombe
intervals (the risk difference is the effect size) and exact McNemar, with Holm across
the comparisons of one family in one world. `*` below = significant after Holm and not
underpowered. Consolidation metrics pool (pair, checkpoint) or (memory, checkpoint)
units, which are not independent, so their Wilson intervals are descriptive.

## Results

### Correct answers (known.correct, scorable known-value probes)

| condition | baseline | short | long | contradiction | drift | repetitive | noisy | adversarial |
|---|---|---|---|---|---|---|---|---|
| A:none | 17/36 | 31/36 | 10/36 | 8/23 | 15/36 | 20/36 | 21/36 | 16/26 |
| B:lossless | 17/36 | 30/36 | 10/36 | 6/23 | 13/36 | 18/36 | 20/36 | 16/26 |
| C:dedup | 17/36 | 30/36 | 10/36 | 6/23 | 13/36 | 18/36 | 20/36 | 15/26 |
| D:abstractive | 17/36 | 30/36 | 10/36 | 6/23 | 13/36 | 18/36 | 20/36 | 16/26 |
| E:hierarchical | **34/36** | 34/36 | **35/36** | **20/23** | 18/36 | **35/36** | **35/36** | 19/26 |
| F:recency | **26/36** | 30/36 | **29/36** | 14/23 | 12/36 | 23/36 | 27/36 | 17/26 |
| G:importance | 16/36 | 26/36 | 9/36 | 2/23 | 13/36 | 16/36 | 17/36 | 14/26 |
| H:hybrid | 24/36 | 33/36 | **22/36** | **18/23** | 17/36 | **30/36** | 26/36 | 18/26 |
| noop | 17/36 | 31/36 | 10/36 | 8/23 | 15/36 | 20/36 | 21/36 | 16/26 |

Bold = significant against A after Holm. The effects (risk difference [95% CI], Holm p):

- **E (temporal) improved correctness** in baseline +0.47 [+0.28, +0.63] (p = 4.6e-5),
  long +0.69 [+0.50, +0.82] (p = 1.8e-7), contradiction +0.52 [+0.26, +0.69]
  (p = 0.0044), repetitive +0.42 [+0.23, +0.58] (p = 1.8e-4) and noisy +0.39
  [+0.19, +0.56] (p = 0.0047). It did not in short (+0.08, p = 1), drift (+0.08, p = 1)
  or adversarial (+0.12, p = 1). The ablations locate the cause in the regime, not in
  abstraction: temporal consolidation without any abstraction (`temporal`,
  `depth:L2`) gives exactly the same answers as E in every world tested, and
  depth L3/L4 add nothing to correctness. **Mechanism:** an episodic memory is valid from
  when it occurred and never ends, so the Phase 6 temporal filter cannot drop a
  superseded value. A temporally consolidated L2 fact carries its period's bounded
  interval, so the filter can.
- **F (recency)** helped in baseline +0.25 (p = 0.031) and long +0.53 (p = 3.1e-5). It
  did not help elsewhere, and in drift it went the other way (−0.08, p = 1). Most probes
  ask about the recent past, and recent promotions add a second, recent-valued copy to
  retrieval. This is sensitivity of this probe design, not evidence that recent memories
  are more true.
- **G (importance) never helped.** It was lower in six of eight worlds, most in
  contradiction (2/23 vs 8/23, −0.26 [−0.47, −0.02], Holm p = 0.42, not significant).
- **Non-temporal dedup (B, C, D) changed no answer significantly** (all Holm p = 1).

Retention was unchanged by every strategy under mixed retrieval, because L1 memories
remain retrievable. The only visible dip was E and H in drift (−0.08, p ≥ 0.25).

### Consolidation fidelity (strategies; pooled over checkpoints)

| world | condition | compression | merge precision | merge recall | abstraction errors | inferred errors | token loss | contradictions kept | stale answers |
|---|---|---|---|---|---|---|---|---|---|
| baseline | B:lossless | 1.95 | 361/515 = 0.70 | 361/750 = 0.48 | — | — | 0/1665 | — | 13/36 |
| baseline | C:dedup | 1.72 | 334/475 = 0.70 | 334/750 = 0.45 | — | — | 0/1795 | — | 13/36 |
| baseline | D:abstractive | 2.08 | 382/536 = 0.71 | 382/750 = 0.51 | 155/263 = 0.59 | — | 32/2024 | — | 13/36 |
| baseline | E:hierarchical | 1.52 | 334/334 = 1.00 | 334/750 = 0.45 | 0/388 | 4/4 | 32/2707 | — | 12/36 |
| baseline | F:recency | 4.78 | 98/98 = 1.00 | 98/225 = 0.44 | — | — | 0/906 | — | 13/36 |
| baseline | G:importance | 4.30 | 322/457 = 0.70 | 322/323 = 1.00 | — | — | 0/745 | — | 13/36 |
| drift | B:lossless | 3.36 | 1274/4696 = 0.27 | 1274/2708 = 0.47 | — | — | 0/3355 | — | 26/36 |
| drift | E:hierarchical | 1.34 | 819/819 = 1.00 | 819/2708 = 0.30 | 0/1258 | 41/41 | 455/8126 | — | 22/36 |
| contradiction | C:dedup | 1.80 | 266/590 = 0.45 | 266/670 | — | — | 0/1980 | 142/142 | 9/36 |
| contradiction | E:hierarchical | 1.72 | 242/311 = 0.78 | 242/614 | 34/466 | — | 0/2714 | 108/142 | 6/36 |
| adversarial | C:dedup | 1.94 | 1167/2303 = 0.51 | 1167/2122 | — | — | 0/3722 | 159/159 | 1/36 |
| adversarial | E:hierarchical | 1.69 | 1167/1634 = 0.71 | 1167/2122 | 18/581 | 53/63 | 590/5644 | 159/159 | 0/36 |

- **"Lossless" is lossless in content, not in meaning.** Exact deduplication lost no
  feature in any world (every loss report empty, as tested). Yet 30% (baseline) to 73%
  (drift) of its merges joined identical statements from *different true periods*: a value
  that recurs is merged across the change in between. Claim dedup does the same. Only
  temporal consolidation reached precision 1.00, in every world without rival or corrected
  reports.
- **Where temporal precision drops below 1**, in contradiction (0.78) and adversarial
  (0.71), the merged reports share key, value and period in the timeline but not the
  generator's true period: a wrong first report and its correction, or a poisoned rival
  value.
- **Abstraction built on open-ended intervals is wrong.** D's entity profiles (claim
  groups with episodic intervals) were wrong in 155/263 statements (baseline) and 686/965
  (long). They listed superseded values as current and marked them "contested". E's
  profiles and timelines erred in 0/388 (baseline) and 34/466 (contradiction).
- **Inferred co-changes were almost all false**: 4/4 (baseline), 41/41 (drift) and 53/63
  in the adversarial world, where one coupling was designed and changes were snapped to a
  30-day grid. Correlation is only inferred here: every such memory is labelled
  `inferred`, carries no claim, and never answers as a fact (tested).
- **Contradiction preservation:** every guarded regime kept both values of every
  same-instant contradiction (142/142, 159/159), except temporal consolidation in the
  contradiction world, 108/142. All 34 unkept cases were contests later resolved by a
  `correct` report: a correction replaces every open value, the Phase 3 standard (checked
  one by one).
- **Lineage completeness was 1.00 in every run**; unsupported derived content was 0 in
  every run. Maximum lineage depth reached 130 (D in drift) and 122 (E in drift).

### Ablations

| world | condition | merge precision | blocked merges | number loss | token loss | contradictions kept | correct |
|---|---|---|---|---|---|---|---|
| baseline | semantic 0.7 | 361/515 | 911 | 0/174 | 0/1665 | — | 17/36 |
| baseline | semantic 0.5 | 382/536 | 2395 | 0/153 | 32/1581 | — | 17/36 |
| baseline | semantic 0.3 | 382/536 | 5766 | 0/153 | 32/1581 | — | 17/36 |
| baseline | semantic 0.3, **no guards** | **361/5256 = 0.07** | 0 | **71/107** | **357/751** | — | 17/36 |
| contradiction | semantic 0.3 | 295/641 | 8535 | 0/151 | 20/1849 | 142/142 | 6/23 |
| contradiction | semantic 0.3, **no guards** | 264/5056 = 0.05 | 0 | 76/132 | 493/998 | **0/142** | 8/23 |
| adversarial | semantic 0.3 | 1309/2525 | 21478 | 0/216 | 53/3107 | 159/159 | 16/26 |
| adversarial | semantic 0.3, **no guards** | 1332/21541 = 0.06 | 0 | 115/169 | 693/1443 | **0/159** | 16/26 |

With the guards on, lowering the similarity threshold from 0.7 to 0.3 multiplied
*refused* merges (911 → 5766 in baseline) while barely changing *accepted* ones. Without
the guards, the same threshold merged unrelated facts: merge precision 0.05–0.07,
58–68% of numbers lost, every same-instant contradiction collapsed, and derived memories
that almost never answered (0–3/36). The final answers barely changed: 17/36, 8/23 and
16/26 against 17/36, 6/23 and 16/26 with guards, because mixed retrieval still reaches the
untouched L1 evidence. Consolidation damage is a memory-integrity finding that answer
accuracy alone can hide.

**Importance leave-one-out** (claim regime, threshold 0.5, contradiction world; correct
answers against A's 8/23): dropping contradiction 11/23, repetition 6/23, provenance
4/23, persistence 2/23, recency 2/23, source 1/23; all six components give 2/23. No
difference survives Holm (smallest adjusted p = 0.16). A unit test shows the
decomposition exposing the exploit directly: three unreliable copies of a value outrank
one reliable report on repetition. The equal-weight model promotes the copies and archives
the reliable report.

### Stability–plasticity

| world | consolidate every | correct | stale answers | hierarchy KiB | run s |
|---|---|---|---|---|---|
| baseline | 7 d | 36/36 (+0.53*) | 3/36 | 3797 | 4.4 |
| baseline | 30 d | 34/36 (+0.47*) | 12/36 | 949 | 3.1 |
| baseline | 90 d | 30/36 (+0.36*) | 12/36 | 362 | 2.7 |
| drift | 7 d | 34/36 (**+0.53\***) | 4/36 | 13671 | 29.6 |
| drift | 30 d | 18/36 (+0.08) | 22/36 | 3034 | 15.7 |
| drift | 90 d | 9/36 (**−0.17**, p = 1) | 27/36 | 817 | 10.1 |
| long | 7 d | 35/36 (+0.69*) | 2/36 | 16884 | 19.8 |
| long | 90 d | 28/36 (+0.50*) | 11/36 | 1313 | 6.6 |

The trade-off is measured, not assumed. In the fast-changing drift world, consolidating
weekly produced one of the largest gains in the study (+0.53 [+0.33, +0.68], Holm
p = 5.3e-5). Consolidating every 90 days fell *below* no consolidation (9/36 vs 15/36):
27 of 36 answers came from derived memories whose evidence had changed after they were
built. Plasticity costs storage and time: weekly consolidation wrote 4× (against 30 days) to
17× (against 90 days) the hierarchy bytes and took 1.4–2.9× the run time. Recency windows (30/90/180 d) helped
in baseline and long, and not in drift. Importance thresholds 0.3/0.5/0.7 compressed up to
26.6× (long, 0.7) with no significant change in answers.

### Retrieval over the hierarchy

| world | raw only (L1) | mixed (L1–L4) | consolidated only (L2–L4) | hierarchy-aware (mixed + support, stale excluded) |
|---|---|---|---|---|
| baseline | 17/36; retention 36/36 | 34/36; 36/36 | 33/36; 33/36 | 27/36; 36/36 |
| long | 10/36; 36/36 | 35/36; 36/36 | 35/36; 35/36 | 27/36; 36/36 |
| drift | 15/36; 35/36 | 18/36; 32/36 | 18/36; **20/36** | 19/36; 31/36 |
| repetitive | 20/36; 36/36 | 35/36; 36/36 | 31/36; 31/36 | 33/36; 36/36 |

(all with hierarchical consolidation; raw only ≡ no consolidation.) Consolidated-only
retrieval cannot see anything learned since the last checkpoint. In drift its retention
fell by −0.42 [−0.58, −0.23] (Holm p = 1.8e-4) against raw. Excluding stale derived
memories removed stale answers (0/36 everywhere) and restored retention. It also
answered less often correctly than mixed retrieval in baseline and long (27/36 vs
34/36 and 35/36). The added support signal favours large, older groups.

### End-to-end demonstration (baseline world; `demonstration()`)

1. **Formation → consolidation → retrieval → evaluation.** Hierarchical consolidation
   with mixed retrieval answered 34/36 known-value probes correctly (run `sha256:2112d567…`).
2. **Provenance traversal.** Probe `ana.dose-0` was answered `ana.dose = 10 mg` from the
   derived L2 memory `L2:7abb3c17a3caeb2ff407` (status derived). Lineage: that memory →
   version `sha256:b0bc0d20…` → experience `chat:baseline-4`.
3. **Information loss.** Under D (abstractive), the worst loss report is an L3 entity
   profile whose content dropped eight entity mentions (Acme, Berlin, Initech, Oslo, Rome,
   Umbrella, Vandelay, Vienna). They are historical values its input facts mention but a
   current-state profile does not, still reachable through its lineage.
4. **Adversarial perturbation and re-consolidation.** The Phase 3 `contaminate`
   intervention (50% of statements, same-instant false copies) was applied and
   re-consolidated. Correct answers went 34/36 → 32/36 (comparison `sha256:df36cbf4…`,
   variable `interventions`).
5. **Efficiency.** In the repetitive world, a consolidated-only query considers 179
   memories instead of 287 (−38%). Median retrieval latency was unchanged (≈14 ms both,
   environment-bound). The saving is in candidates considered, not in time at this scale.

## What this shows, and what it does not

- Under these generated worlds, **the consolidation regime mattered more than its
  depth.** Occurrence-ordered temporal consolidation raised correct answers in five of
  eight worlds. Deduplication, abstraction and importance-based promotion did not
  significantly raise them anywhere.
- **Consolidation damage can be invisible in answers.** Unguarded semantic merging
  destroyed merge precision, numbers and contradictions, and left answers almost
  unchanged, because the evidence stayed retrievable. Memory-integrity measurements are
  needed in addition to accuracy.
- **Stability costs plasticity.** Infrequent consolidation of a fast-changing world turned
  a gain into a loss through stale derived memories.
- It does **not** show that any strategy is better in general. Worlds are generated with
  one seed and one statement language; 36 probes per world make most comparisons
  underpowered; guards and features are surface heuristics; the embedder is the lexical
  reference, not a semantic model.
