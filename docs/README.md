# MEMORIA documentation

| Document | Read it for |
|---|---|
| [../README.md](../README.md) | Overview, findings with scope, installation, quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Research questions, principles, invariants (I1–I89), semantics per layer, module boundaries, storage, decision log |
| [ROADMAP.md](ROADMAP.md) | Phase-by-phase scope, exit criteria and honest status |
| [examples/lifecycle-demo.md](examples/lifecycle-demo.md) | Output of `python -m memoria.lifecycle_demo` (a demonstration, not a benchmark) |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Ground rules and the quality gate |

## Experiment reports

Each report states its hypotheses, definitions, method, results with uncertainty, negative findings
and limitations, and the digests needed to reproduce it.

| Report | Study |
|---|---|
| [phase5-semantic.md](experiments/phase5-semantic.md) | Lexical-subword vs neural representation |
| [phase6-hybrid.md](experiments/phase6-hybrid.md) | Hybrid, explainable retrieval |
| [phase7-consolidation.md](experiments/phase7-consolidation.md) | Consolidation and stability–plasticity |
| [phase8-10-graph-forgetting-interference.md](experiments/phase8-10-graph-forgetting-interference.md) | Memory graph, forgetting, interference |
| [phase11-12-beliefs-calibration.md](experiments/phase11-12-beliefs-calibration.md) | Belief revision and calibration |
| [superphase5-adaptive-memory.md](experiments/superphase5-adaptive-memory.md) | Adaptive importance, retention, free-text claims |
| [superphase6-autopsy-replay-benchmark.md](experiments/superphase6-autopsy-replay-benchmark.md) | Autopsy, replay, counterfactuals, multi-seed benchmark |

The historical studies were not re-run for the release; the release checks re-execute only the
deterministic demonstrations and the test suite.

## Diagrams

Sources for the README figures are in [assets](assets): `hero.svg`, `lifecycle.svg`,
`architecture.svg`, `autopsy-flow.svg`, and `social-preview.svg` with its 1280×640 export `social-preview.png`. GitHub's social preview cannot be
set through the CLI or API; upload the PNG in the repository settings (Settings → General → Social preview).
