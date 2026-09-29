# Super-Phase 5: adaptive importance, retention dynamics, free-text claims and entity identity

Run with `python -m memoria.adaptive_run var/superphase5` (`--quick`: 3 replicates, two
worlds). Every number below is read from the full run: 5 worlds x 24 systems x 12 replicates =
1,440 stored runs, plus 6 calibration replicates for two systems. Study digest
`sha256:558cdbba…0bcb`, spec `sha256:8217bf2e…eb37`. Reproduction: the runs of replicates 0
and 11 (240 runs) were re-simulated into an independent store and all 240 digests are identical;
the test suite also checks that two executions of a smaller study give the same study digest.
Trace audit: 15 traced runs, 0 cutoff violations, 0 re-derivation violations, 0 broken links.
The full report (all tables) is printed by the run; this document keeps the tables that carry
the conclusions.

## Research questions and hypotheses

Design decisions (the lambda and theta doses, the `age`-based static baseline) were made after
a pilot of 3 replicates in 2 worlds; no pilot number appears below, but the pilot showed the
direction of the entrenchment effect, so H2 is not a blind prediction. Hypotheses were fixed
before the full run:

- **H1** Importance derived from use predicts *future use* (AUROC well above 0.5) but is only a
  weak predictor of *truth*; the two must be reported apart.
- **H2** Letting usage-driven importance influence ranking (lambda > 0) eventually entrenches
  stale values in a changing world; there is a dose beyond which stale answers rise.
- **H3** Importance-gated retention can raise stale answers, but only at high selectivity
  (theta); low selectivity leaves ranking, not retention, as the cause.
- **H4** Supersession-aware retention reduces stale answers in changing worlds and is defeated
  by a stale-value flood ("newer" is not "truer").
- **H5** Adaptive importance beats a static (age + provenance) heuristic.
- **H6** Conservative extraction never turns uncertain text into fact; identity by similarity
  produces false merges the conservative rule does not.

## Definitions

- **Item**: one extracted, entity-resolved claim about a key (`entity.attribute`), with its
  report, source and claim id. **Archived** = not retrievable; nothing is deleted.
- **Audit probe**: every 10 days per key, the system's current answer, judged against the world's
  hidden truth. Probes create no events (I83). Probes before day 30 are excluded.
- **Outcomes**: `correct`; `stale` (the answer was true earlier for that key); `wrong` (never
  true); `none` (no retrievable item).
- **Retrieval retention** is reported as the correct-answer rate and `none` rate; **evidence
  retention**: of probes where a recorded item supports the current truth, the share where a
  retrievable one does; **item retention**: retrievable / recorded items; **revision latency**:
  days from a true change until an audit probe answers the new value (censored at the end of the
  period; the censored share is reported). Resolution is 10 days.
- **lambda** in `(1 - lambda) * freshness + lambda * importance`: 0 is "latest occurrence
  wins". **theta**: importance threshold of the gate on the declared importance scale (an unused
  recorded item scores about 0.42).
- Systems A-K and the doses R (lambda 0.25/0.75/1) and T (theta 0.45/0.55/0.6/0.65, lambda 0.5),
  and X (one importance component removed from F), are listed in the run's Systems table.

## Design and statistics

Five generated worlds: stable (no true changes), changing (a change about every 45 days),
repetitive (skewed query popularity, duplicated reports), noisy (25% stray reports, hedged
and pronoun text, aliases, typos) and adversarial (8 entities with near-collision names, 3 target
keys flooded with 12 stale-value reports after a change and pumped with 40 queries before it).
Reports are text (60-80% free text) with hidden gold labels. Replicates are independent seeds.
Inference is per replicate: paired difference against a named baseline, Student-t interval, exact
sign test, Holm within a family (one world x one metric x the listed comparisons), underpowered
flag. Proportions are pooled over probes with Wilson intervals (probes of one key are
correlated; not corrected). "Worse/better" means Holm-adjusted p < 0.05, not underpowered.
With 12 replicates the smallest reachable Holm-adjusted p in a 20-comparison family is 0.0107,
so a comparison that is not significant here may still be a real effect.

## Results

### Importance: use is predictable, truth much less (H1)

Importance is fitted-free (declared weights); only its *calibration* is fitted, on separate
replicates (I74).

| world | system | P(used again in 30 d) | AUROC use | AUROC truth | ECE raw -> recalibrated |
|---|---|---|---|---|---|
| stable | A (passive) | 0.164 | 0.963 | 0.672 | 0.403 -> 0.011 |
| stable | B (loop active) | 0.169 | 0.994 | 0.710 | 0.404 -> 0.006 |
| changing | A | 0.063 | 0.828 | 0.782 | 0.447 -> 0.006 |
| changing | B | 0.070 | 0.914 | 0.771 | 0.435 -> 0.004 |
| repetitive | A | 0.029 | 0.831 | 0.781 | 0.477 -> 0.003 |
| repetitive | B | 0.032 | 0.910 | 0.786 | 0.462 -> 0.004 |
| noisy | A | 0.133 | 0.815 | 0.713 | 0.406 -> 0.009 |
| noisy | B | 0.138 | 0.877 | 0.703 | 0.398 -> 0.011 |
| adversarial | A | 0.297 | 0.918 | 0.779 | 0.280 -> 0.015 |
| adversarial | B | 0.300 | 0.955 | 0.794 | 0.288 -> 0.016 |

Importance ranks future use very well (AUROC 0.82-0.99) and is a **ranking score, not a
probability** (raw ECE 0.28-0.48; a monotone map fitted on other replicates brings it to
0.003-0.016). Its AUROC for *truth* is lower in the stable world (0.67-0.71) and 0.70-0.79
elsewhere: it is informative, but this is largely because provenance and freshness enter it, not
because use implies truth. H1 is supported in these worlds. AUROC for use is higher when the loop
is active (B) than when importance is only observed (A) in every world: part of the
predictability is the loop feeding itself, which is why passive scoring (A) is the number to
quote.

### Retention curves and what forgetting costs

Changing world, full horizon (pooled probes, Wilson 95%):

| system | correct | stale | no answer | evidence retention | item retention | revision latency (days) |
|---|---|---|---|---|---|---|
| A static, keep all, latest | 0.610 | 0.270 | 0.000 | 1.000 | 1.000 | 19.2 [18.2, 20.3] |
| B adaptive ranking (lambda 0.5) | 0.666 | 0.268 | 0.000 | 1.000 | 1.000 | 19.1 |
| C age window 30 d | 0.576 | 0.215 | 0.101 | 0.826 | 0.148 | 19.3 |
| D age window 90 d | 0.610 | 0.270 | 0.000 | 0.936 | 0.423 | 19.2 |
| E gate 0.5, latest | 0.612 | 0.272 | 0.001 | 0.919 | 0.419 | 19.2 |
| F gate 0.5, lambda 0.5 | 0.666 | 0.269 | 0.001 | 0.908 | 0.374 | 19.1 |
| G stale-aware gate, lambda 0.5 | 0.643 | 0.257 | 0.000 | 0.811 | 0.137 | 18.5 |
| H static importance, lambda 0.5 | 0.655 | 0.271 | 0.000 | 1.000 | 1.000 | 19.1 |

Stale-answer rate over time (mean over replicates; full intervals in the run): A 0.17 (day 60),
0.22 (90), 0.27 (180), 0.37 (300), 0.24 (360). Evidence retention of the 30-day window falls from
1.00 (day 30) to 0.89 (60) and 0.77-0.80 (300-360).

- Forgetting is not free and does not fix staleness by itself. In four of five worlds the
  30-day window lowers the stale rate (changing -0.055 [-0.064, -0.046], noisy -0.128) but pays
  in answers: correct falls by 0.034 (changing), 0.093 (noisy), 0.518 (adversarial) and 0.726
  (stable), with 2-88% of probes unanswerable. The stale rate falls because the answers
  disappear (`none` 0.101 in changing), not because they become right.
- The importance gate (F) keeps 37-43% of items while keeping 91-93% of the evidence for the
  current truth (changing, adversarial); the 90-day window keeps 42% of items and 94% (changing).
- Stale-aware retention (G) archives the most (item retention 0.137, evidence retention 0.81)
  and buys a small gain in stale rate (-0.012 [-0.021, -0.004] against F, not significant after
  Holm) at a cost in correct (-0.023, Holm p 0.07).

### Where retention starts causing stale-memory damage (H2, H3)

*Ranking dose (keep everything, vary lambda; baseline A, stale-rate difference, Holm within the
dose family):*

| world | lambda 0.25 | 0.5 | 0.75 | 1.0 | onset |
|---|---|---|---|---|---|
| changing | -0.009 | -0.001 | **+0.257** | **+0.592** | 0.75 |
| repetitive | -0.003 | +0.002 | **+0.115** | **+0.496** | 0.75 |
| noisy | -0.008 | -0.012 | **+0.169** | **+0.372** | 0.75 |
| adversarial | -0.002 | **-0.033** | **-0.075** | -0.012 | none (lower stale) |
| stable | 0 | 0 | 0 | 0 | none (no changes) |

Bold: Holm p < 0.05. At lambda = 1 (importance only) the changing world answers correctly on
9.3% of probes and 86% are stale; 94.5% of true changes are never reflected in an answer. The
time of onset for lambda = 1 against A: day 60 (changing), day 90 (repetitive, noisy). H2 is
supported: usage-driven importance entrenches the old value once it is allowed to outweigh
freshness (between lambda 0.5 and 0.75 here). This is consistent with the old item having
accumulated uses and confirmations before the change; the study measures the effect, not that
mechanism directly.
The adversarial world is the exception: importance, through provenance, lowers the stale rate
because the stale flood comes from a low-prior source class.

*Retention dose (gate threshold theta at lambda 0.5; baseline B = keep everything under the same
ranking, so only the schedule changes):*

| theta | item retention (changing) | Delta stale changing | noisy | repetitive | adversarial |
|---|---|---|---|---|---|
| 0.45 | 0.513 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.374 | +0.001 | +0.002 | +0.002 | -0.002 |
| 0.55 | 0.288 | +0.001 | +0.010 | +0.002 | **-0.024** |
| 0.60 | 0.227 | **+0.008** | **+0.024** | -0.005 | **-0.038** |
| 0.65 | 0.182 | +0.016 (p 0.15) | **+0.035** | **-0.025** | **-0.042** |

Retention itself begins to cause stale damage at theta = 0.60 in the changing world (item
retention 0.23; +0.008, small) and in the noisy world (retention 0.29; +0.024); it never does in
the repetitive world (it *reduces* stale answers at 0.65) or the adversarial world (it removes the
flood). H3 is supported but the effect is an order of magnitude smaller than the ranking dose:
in these worlds what harms is what feedback does to ranking, not what the gate archives. Onset
over time for the gate against B: none in any world; supersession-aware gate against B: day 240
in the adversarial world (+0.045, rising to +0.062 by day 360), the flood defeating the
supersession rule. **H4**: in the changing worlds the stale-aware gate lowers the stale rate only
slightly and not significantly (changing -0.012 against F, repetitive -0.007, noisy -0.002), so
the first half is not supported; the second half (defeated by a flood) is supported.

### Adaptive against static (H5)

B/F raise the correct rate over the static baseline A in four worlds (changing +0.056, repetitive
+0.029, noisy +0.065, stable +0.141, all Holm p 0.0107), and the stale-rate change is not
significant. Against the event-free heuristic H (age + provenance) the gain is small and mostly
not significant: F - H correct +0.011 (changing, p 0.05), +0.015 (repetitive, p 0.33), +0.032
(noisy, p 0.08), 0.000 (stable), -0.006 (adversarial). **Most of the benefit of "importance" here
comes from source provenance, not from usage feedback.** H5 is not supported beyond what the
provenance component provides.

### Ablation (system F, change against the full model)

| removed | changing: stale / correct | noisy | repetitive | adversarial |
|---|---|---|---|---|
| correction | **+0.090 / -0.092** | **+0.043 / -0.042** | **+0.026 / -0.017** | -0.006 / -0.014 |
| provenance | +0.014 / **-0.036** | +0.002 / -0.011 | -0.009 / -0.006 | **+0.021 / -0.056** |
| recency | **+0.024** / -0.008 | **+0.025** / +0.006 | **+0.015 / +0.012** | **-0.038 / +0.064** |
| frequency | -0.013 / +0.009 | +0.001 / +0.009 | -0.005 / -0.003 | -0.015 / **+0.047** |
| success | -0.003 / +0.005 | -0.002 / +0.002 | 0.000 / +0.004 | **-0.017** / +0.004 |
| contradiction | +0.003 / -0.001 | +0.002 / +0.004 | -0.003 / -0.001 | +0.009 / -0.014 |

The correction component (a source retraction or user correction lowers an item) carries the most
useful signal against staleness in all three changing worlds. Provenance matters for correct
answers where sources differ. Frequency and success add nothing detectable or slightly hurt:
removing frequency *improves* the adversarial world by 0.047, because pumped queries inflate it.
Every component has a mechanism, none is tuned; weights are declared, and a different weighting
would give different numbers.

### Feedback timing

Recording user feedback 7 days late (K against F): correct -0.008 in changing (Holm p 0.0107),
-0.007 in noisy (p 0.08), no detected change elsewhere; stale +0.006 (changing, p 0.12). Late
feedback is measurably, but slightly, worse, and the ledger's cutoff makes the difference visible
instead of leaking.

### Extraction, unresolved claims and identity (H6)

| world | identity | reports | precision | recall | uncertain -> fact | unresolved reports |
|---|---|---|---|---|---|---|
| stable | conservative | 864 | 1.000 | 0.983 | n/a | 15 |
| changing | conservative | 3974 | 1.000 | 0.987 | n/a | 51 |
| repetitive | conservative | 6277 | 1.000 | 0.984 | n/a | 98 |
| noisy | conservative | 2770 | 1.000 | 0.930 | 0 (Wilson upper 0.009) | 581 (hedged 256, pronoun 159, near-name 166) |
| adversarial | conservative | 1123 | 1.000 | 0.903 | 0 (upper 0.066) | 158 (ambiguous 54, near-name 104) |
| noisy | similarity | 2770 | 1.000 | 1.000 | 0 (upper 0.009) | 415 |
| adversarial | similarity | 1123 | 1.000 | 1.000 | 0 (upper 0.066) | 54 |

The conservative rules extracted no uncertain text as fact and no claim with a wrong entity,
attribute, value or kind; every extracted claim's spans reproduce its text. The price is
recall (0.90-0.99) and 0-21% unresolved reports; the unresolved claims stay as objects with a
reason. Downstream, gold ingestion (I) and conservative extraction (A) differ by at most 0.007
in the correct rate (changing 0.612 vs 0.610, noisy 0.430 vs 0.424, adversarial 0.587 vs 0.583;
stable 0.816 vs 0.823, where gold is lower): the extraction loss is small next to the effects
above, and it is not tested for significance here. **False merges: 0 under all three identity policies in all
generated worlds**, because the worlds never mention an unregistered person with a name near a
registered one. The rules that produce false merges are therefore shown only by the designed
cases: the `similarity` policy resolves "Benno" (unregistered) to Benn (false merge) where the
conservative and exact-only policies do not; two independent sources asserting Ana = Ben with no
declared distinction merge (flagged false against the designed truth), while Ana = Anna is a
`conflict` because the registry declares them distinct. 56/56 designed cases (28 extraction,
24 identity, 4 merge) pass, and they are checks of the rules, not evidence about real text. H6
is supported for the first half; the second half is demonstrated, not measured on the worlds.

### Cost and latency

Deterministic proxies: retrievable items per probe fall from 13.6 (keep all) to 5.1 (gate) and
1.9 (stale-aware) in the changing world; importance evaluations rise to 166k-217k per 12 runs
for adaptive ranking against 0 for static; items scanned per decision are unchanged (34). Wall
time (a separate `PerformanceRecord`, not part of any digest): 5 s (A) to 16-18 s (adaptive
gate) summed over 60 replicate-worlds each, so adaptive ranking costs 2.6-3.6x the static
baseline in this pure-Python implementation.

### Provenance completeness

Every answered probe resolved from its item to the report text and, for free-text claims, to a
span that reproduces the value, in every world and system (Wilson lower bound 0.994 or higher).
This holds by construction and the number verifies that no run broke it. The trace audit re-derived all 15 traced runs' decisions and importance records without violations.

## Failure cases and negative findings

- Importance-only ranking is a failure by construction (lambda = 1: 86% stale in the changing
  world); it is measured to locate the onset, not proposed.
- The stale-aware rule is defeated by the flood: superseded-by-newer archives the true item when
  the flood is newer (evidence retention 0.73 in the adversarial world).
- Age windows trade staleness for missing answers; at 30 days the stable world answers 10% of
  probes.
- Frequency and success feedback are not detectably useful here; pumped access inflates
  frequency (adversarial).
- No false merges arose in the generated worlds (see above).

## Limitations

Worlds, declared weights, thresholds and doses are generated or declared, not tuned or real; the
simulated user is a model (its noise, 50% error noticing, is a declared parameter); text comes
from templates and the extractor is a small rule table; probes of one key are correlated (only
replicate-level inference is corrected); 12 replicates bound the smallest reachable p-value;
revision latency has 10-day resolution and censoring; retention decisions are recomputed per
probe (O(items) per decision); the lab is not a run-manifest variable; nothing here says what
happens with real users or real text.

## Deferred

Adaptive retrieval policies inside the hybrid pipeline and their manifest variable (Phase 13);
the same curves on Phase 15 benchmarks, incremental state, capacity and contamination curves
(Phase 16); unregistered near-name mentions in the worlds so that false merges can be measured
rather than only demonstrated; learned or local-model extraction behind the same claim records;
autopsy over retention decisions (Phase 17).
