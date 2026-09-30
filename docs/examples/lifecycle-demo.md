# MEMORIA lifecycle demonstration

> **This is a demonstration, not a benchmark.** It follows one generated world through
> the pipeline, chosen by fixed rules, to make the stored artifacts inspectable. Its
> numbers are single-world illustrations; the measured, multi-seed results are in
> `docs/experiments/`.

## 1. Experience → memory → retrieval → contradiction
- world: `contradictory`; stages (run digests): graph-l1=`sha256:3a84509b72fe`, graph=`sha256:6bf2c00624a7`, forgetting=`sha256:cf19082f6b5b`, interference=`sha256:e5908075c3ca`
- contradiction: probe `ana.employer-0` answered from `sha256:8793c6228cc2`; the graph shows 2 contradiction edge(s), retrieval surfaced 9 counter-memory(ies), and 1 contradicting memory(ies) were visible only in the graph

## 2. Consolidation, forgetting and interference
- graph before / after targeted forgetting: `sha256:a654bd8386d8` / `sha256:80c09869e8e4`; diff `sha256:bd960b658e3f`
- forgetting is an availability intervention (nothing is deleted): retention 186/270 = 0.689 [0.631, 0.741] (95% Wilson); precision 48/48 = 1.000 [0.926, 1.000] (95% Wilson); recall 48/48 = 1.000 [0.926, 1.000] (95% Wilson)
- probes whose answer changed under forgetting: 12
- wrong-memory + contaminated answers before / after interference injection: 4 / 7
- failures introduced / reduced across all stages: 9 / 2

## 3. Provenance of an answer given from a derived memory
- graph `sha256:c09c85741c22`; walk from `retrieval:sha256:e91f29f700c5c41eed5d7d9`: 17 steps reaching 2 experience(s), 2 source(s), 3 claim(s), 1 consolidation(s)

## 4. Feedback, importance, retention → autopsy
- adaptive world `sha256:5e7075d0d874`; memory autopsy of `benn.employer` at day 10.5, answer 'umbrella':
  answer:found, decision:found, candidates:found, policy:found, signals:found, selected:found, state:found, resolution:found, experience:found, source:found, validity:found, corrections:found, contradictions:found; missing=none; complete=True

## 5. Replay around a source correction
- key `chen.home`: state of item `item:r:chen.home:0:1` just before / after correction `event:73f9ad4b33f137eb` (recorded at day 40.4954): ('active', 'importance_above') → ('archived', 'source_corrected')
- snapshot digests before / after: `sha256:af55d9d23aab` / `sha256:d5b7fdeab6f3`

## 6. Counterfactual replay on fixed evidence
- F: rank weight 0.5 -> 1.0: 767/1600 recorded decisions change (767/1600 = 0.479 [0.455, 0.504] (95% Wilson)); gained 8, lost 180 (analysis only, against hidden truth). decision-level recomputation with the recorded evidence held fixed; not a causal estimate of system behaviour under the alternative, whose own feedback events would differ
- B -> F: keep all -> gate 0.5: 1/1600 recorded decisions change (1/1600 = 0.001 [0.000, 0.004] (95% Wilson)); gained 0, lost 0 (analysis only, against hidden truth). decision-level recomputation with the recorded evidence held fixed; not a causal estimate of system behaviour under the alternative, whose own feedback events would differ
- F: remove correction: 59/1600 recorded decisions change (59/1600 = 0.037 [0.029, 0.047] (95% Wilson)); gained 1, lost 7 (analysis only, against hidden truth). decision-level recomputation with the recorded evidence held fixed; not a causal estimate of system behaviour under the alternative, whose own feedback events would differ

## 7. Belief revision and calibrated uncertainty → belief autopsy
- `B:source-weighted` on `ana.dose` at day 133.491, answer '30 mg':
  answer:found, decision:found, policy:found, candidates:found, signals:found, selected:found, state:found, resolution:found, experience:found, source:found, validity:found, corrections:found, contradictions:found; missing=none; complete=True

## 8. Verification
- artifact store re-hashed: ok; memory-lifecycle rebuilt into an independent store: identical; adaptive-lifecycle rebuilt: identical
- digests: memory lifecycle `sha256:a7a87c4819accbf1980135db6271c6e3c8c392c884a0f55305fa8bf0b243dc97`; adaptive lifecycle `sha256:cbdb9096702b252f1205befeb6940c5c1b5b18e792b43ffb57644aaa132a8625`
