# Phases 11–12: contradiction reasoning, belief revision and calibrated uncertainty

Run with `python -m memoria.belief_run var/phase11` (`--quick`: three worlds). The run
prints a report, stores every ledger as an artifact and reproduces all of them (and the
demonstration) in an independent store. All numbers below come from the full run:
10 worlds, 8 policies, 80 ledgers reproduced identically, 0 audit violations, 0 historical
mismatches. Study digest `sha256:0b2e7f95…b7d`. Hidden truth and hidden source reliability are used only for analysis.

## Research questions

1. Can a system keep change, correction and contradiction apart without equating
   recency with truth?
2. Which revision signals (count, declared trust, recency, temporal validity,
   corroboration, contradiction penalty, provenance depth, conservatism) change answers,
   and at what cost in coverage?
3. Is the confidence a belief reports calibrated, for whom does it fail, and does
   abstention prevent errors?
4. How order-dependent is belief revision when the same evidence arrives in another order?
5. Do consolidation, forgetting and interference change confidence without changing
   correctness?
6. Does a belief answer beat top-1 hybrid retrieval on the same probes?

## Definitions

- **Belief**: a stance on one value of one key over a valid-time interval, derived from
  evidence; states SUPPORTED, CONTESTED, CORRECTED, SUPERSEDED, UNKNOWN, UNRESOLVED,
  REJECTED. There is no "false" (REJECTED = "not adopted").
- **Score**: `weight / (competing mass + prior mass)`, a rule-based confidence in [0, 1].
  It is a ranking score until a calibration supports a probabilistic reading; log loss is
  therefore reported only for recalibrated scores.
- **Calibration**: equal-width reliability with Wilson bin intervals; ECE, MCE, Brier;
  isotonic recalibration fitted on calibration replicates and evaluated on test replicates.
- **Overconfidence** = mean score − accuracy (negative = underconfident).
- **Order sensitivity**: the same evidence set arriving in ten different orders.

## Design

Ten generated worlds (stable, temporal-change, contradiction, delayed-correction,
source-disagreement, duplicated-evidence, adversarial, ambiguous-entities, numeric,
long-horizon) with hidden truth and hidden reliabilities; eight policies A–H; test
replicates for evaluation and separate replicates for fitting calibrators. Statistics:
paired Newcombe differences, exact McNemar, Holm within world and family, key-level
cluster analysis (paired t, sign test, d_z), underpowered flags; no resampling.
Comparisons between policies are observational over generated worlds and are not causal
claims about real-world data.

## Results

### Policies pooled over worlds (1504 probes)

| policy | coverage | correct (of all) | selective acc. | ECE raw → recalibrated | Brier | AURC (oracle) |
|---|---|---|---|---|---|---|
| A evidence-count | 0.91 | 0.72 | 0.79 | 0.191 → 0.008 | 0.188 | 0.115 (0.024) |
| B source-weighted | 0.97 | 0.75 | 0.78 | 0.108 → 0.031 | 0.160 | 0.099 (0.027) |
| C recency-aware | 0.72 | 0.57 | 0.78 | 0.260 → 0.007 | 0.231 | 0.143 (0.025) |
| D temporal-validity | 0.88 | 0.73 | 0.82 | 0.234 → 0.025 | 0.190 | 0.095 (0.016) |
| E corroboration | 0.50 | 0.47 | 0.94 | 0.277 → 0.026 | 0.138 | 0.034 (0.002) |
| F contradiction-penalty | 0.87 | 0.69 | 0.79 | 0.183 → 0.021 | 0.182 | 0.112 (0.023) |
| G provenance-depth | 0.91 | 0.72 | 0.79 | 0.193 → 0.017 | 0.187 | 0.109 (0.024) |
| H conservative | 0.45 | 0.43 | 0.95 | 0.210 → 0.019 | 0.092 | 0.028 (0.001) |

Source-weighted revision (B) answers the most probes correctly (0.75 vs 0.72 for
evidence-count) and is the best-calibrated raw score. Recency-aware revision (C) is the
worst: it answers the fewest probes and is correct on 0.57 of them, which is the
"recency is not truth" invariant showing up as a measured cost. Corroboration (E) and the
conservative policy (H) reach a selective accuracy of 0.94–0.95 by answering only 45–50%
of probes. Per-world tables, all paired comparisons with Holm-adjusted p, CIs and effect
sizes are in the printed report (many single-world differences are flagged underpowered).

### Calibration

Raw scores are systematically *under*confident (overconfidence −0.11 to −0.28): the
0.5–0.6 bins are right 66–94% of the time. Isotonic recalibration on separate replicates
brings ECE to 0.007–0.031 on the test replicates. Calibration failures survive
aggregation: ECE is 0.31 (A) and 0.42 (C) for clinic-sourced claims, against 0.19 and
0.28 for chat; it grows with time (A: 0.14 early, 0.19 mid, 0.23 late); and claims with
provenance depth ≥ 2 are badly calibrated (ECE 0.39–0.70, few answers). Which
uncertainty components carry information depends on the policy (AUROC for error: 1 − score
0.67–0.77; epistemic 0.44–0.70; source 0.50–0.69; identity and retrieval 0.50 because no
policy uses them in these worlds). Corroboration's epistemic uncertainty is *below* chance
(0.44) — an anti-informative component, reported as found.

### Abstention

Abstaining is worth it for A, D, F, G (net +47 to +69 errors prevented) and costly for C,
E and H (net −139 to −345: they give up many correct answers). B abstains rarely (47) and
nets +3.

### Order sensitivity

A, E, F, G converge to identical final beliefs in every arrival order (100/100); B, C, D
do not (10/100), because learned trust, recency and staleness depend on when evidence
arrives; H differs in 15/100. Final *answers* nevertheless agree 0.94–1.00, while
mid-horizon agreement is 0.84–0.92 and correctness varies widely between orders
(C: 0.31–0.83). Historical reconstructions match the live runs in every order (0
mismatches).

### Stability

Corroboration (E) and conservative (H) churn least (5.6–5.8 vs 12.8–20.7 changes) and
oscillate least, at the price of persistent conflict (0.60–0.62 conflict persistence,
30 days to resolve, 41–56 unresolved at end). Recency (C) churns and oscillates most.

### Learned source trust

Learned trust tracks the hidden reliability well in benign worlds (Spearman 0.80–1.00,
MAE 0.05–0.13) and poorly in the adversarial world (Spearman 0.47, MAE 0.135): the
"trusted" email source is ranked worst there (0.33). It is circular by construction and is
always labelled so.

### Contradiction taxonomy and survey

All 19 designed cases are classified as designed and all twelve types are exercised. In
the worlds, every relation read as genuine involved an erroneous report (precision ≈ 1.00;
0.90 in ambiguous-entities) and no relation between two correct reports was read as
genuine except 15/1641 in ambiguous-entities (identity collisions). This precision is
partly by construction, since the generated wrong reports are the only source of
concurrent disagreement; it is a check of the type rules, not of real text.

### Adversarial cases

Policies without corroboration are overconfident on a weak flood and on copied evidence;
corroboration (E) and conservative (H) hold back on copies and misleading summaries
("competing"), and identity variants abstain on entity collisions. A lineage-aware policy
abstains on a summary that weakened a fact instead of trusting it (`weakened-derived
+lineage`: `require-evidence` for all eight). Reputation poisoning is not defeated by any
policy (all answer `rome`, score 0.46–0.50) — a negative finding.

### Memory operations

Consolidation that only adds derived items changes correctness by ≤ ±1 probe;
consolidation that *replaces* raw items lowers correctness (A+lineage 0.65 vs 0.71) and
is the only condition where lineage-aware policies are hurt. Interference lowers
correctness (A 0.67 vs 0.71, B 0.71 vs 0.75). Forgetting changes little. No condition
reproduced the "more confident, not more correct" pattern at the pooled level.

### Belief against retrieval

On the same probes, belief answers are correct on 0.50–0.92 per world against 0.00–0.25
for top-1 hybrid retrieval (exact McNemar p ≤ 0.008 in every world). Retrieval's raw
score is nearly uninformative about correctness (AUROC 0.32–0.68) while the belief score
is 0.53–1.00. The retrieval baseline here is deliberately naive (one candidate, no
value normalisation), so this measures the gap between ranking and belief, not a bound
on retrieval.

### End-to-end demonstration

See the printed report's Demonstration section: a legitimate change (ben.home) recorded
as supersession and not counted as a contradiction; a genuine contradiction
(ana.employer, categorical, unresolved) left open; an evidence event that moves a belief
(`conflict_resolved`, munich and vienna rejected, prague supported with score
0.24 → 0.48); the same key under another arrival order (the score trajectories differ by up to 0.61); a calibration failure found
(subgroup `time:mid` ECE 0.36 against 0.23 overall); an overconfident answer prevented by
abstention (evidence-count answered "38 mg" at 0.50 for a true 40 mg; the conservative
policy chose `competing`); and a trace from the answer back to the original experience. The demonstration
digest is reproduced in an independent store.

## Negative findings and limitations

- Generated worlds with declared sources; real-world reliability and free text are out of
  scope (assertions come from a declared grammar).
- Confidence is a rule-based score; the calibrated value is a monotone map fitted on
  generated replicates and does not transfer to other worlds.
- Reputation poisoning, and evidence from a trusted but wrong source, defeat all policies.
- Learned trust is circular and fails under adversarial sources.
- The high contradiction precision partly reflects how the worlds create disagreement.
- The uncertainty components for identity and retrieval are uninformative here (AUROC 0.50).
- Many per-world paired comparisons are underpowered; pooled effects are stated with CIs.

## Deferred

Adaptive use of confidence in retrieval (Phase 13), attacks on beliefs beyond the ten
adversarial cases (Phase 14), free-text claim extraction, autopsy over belief traces
(Phase 17).
