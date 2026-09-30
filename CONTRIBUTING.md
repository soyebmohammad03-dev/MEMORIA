# Contributing to MEMORIA

Contributions that make the experiments more correct, reproducible or better documented are
welcome. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first: its invariants (I1–I89) are the
project's contract, and the decision log records what was tried and rejected.

## Setup

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Quality gate (must pass; CI runs the same)

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy            # strict, includes tests
uv run pytest
```

Tests that need the pinned neural model skip with a stated reason when it is absent; nothing
downloads implicitly.

## Ground rules

1. **Evidence or silence.** A claim in code, docs or a report must trace to a stored artifact or a
   test. State scope and uncertainty; never present a benchmark on generated worlds as universal.
2. **Do not weaken an invariant.** Historical digests are pinned by golden tests; a change that
   moves a stored digest needs a decision-log entry and a migration note, not a silent update.
3. **Derived is not evidence.** Consolidations, graphs, beliefs and importance never replace the
   log. Interventions perturb datasets or availability, never stored records.
4. **Determinism.** No wall-clock time, `random`, or iteration over unordered collections in
   anything that feeds a digest. Use the SHA-256 `unit_interval` for seeded choices.
5. **Ground truth stays out of the evidence chain.** It may appear only in analysis.
6. **Tests assert scientific behaviour** (what the system must never do or must reproduce), not
   implementation details.
7. **Report negative results.** A study that finds no effect is a result.

## Changes to studies

If you change code that affects a study's numbers, re-run that study, update its report in
`docs/experiments/`, and say what changed and why. Do not edit numbers by hand.

## Pull requests

Keep them focused; describe the behaviour change and how you verified it; update the docs the
change touches. Use the pull-request template.

## Conduct and security

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
