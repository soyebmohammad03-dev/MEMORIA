"""Markdown report for the Super-Phase 6 benchmark. Every number is read from stored records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from memoria.autopsy import Autopsy
from memoria.autopsy_demo import AutopsyDemonstration
from memoria.benchmark import (
    BELIEF_POLICIES,
    ENVIRONMENTS,
    MEMORY_SYSTEMS,
    BenchmarkManifest,
    BenchResult,
)


def f(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join([*out, *("| " + " | ".join(r) + " |" for r in rows)])


def _autopsy_text(a: Autopsy, limit: int = 6) -> list[str]:
    out = [
        f"`{a.subject}` autopsy of `{a.key}` at day {a.at:g}: answer `{a.answer}`, "
        f"complete={a.complete}, missing={list(a.missing)}"
    ]
    for link in a.links:
        detail = "; ".join(f"{k}: {v}" for k, v in link.detail[:limit])
        out.append(f"- **{link.step}** [{link.status}] {detail}")
    for link in a.after_cutoff:
        detail = "; ".join(f"{k}: {v}" for k, v in link.detail[:3])
        out.append(f"- *after the cutoff* **{link.step}** {detail}")
    return out


def report(m: BenchmarkManifest, r: BenchResult, demo: AutopsyDemonstration) -> str:
    cells = {
        (c.track, c.environment, c.system): {n: (p, i) for n, p, i in c.metrics} for c in r.cells
    }
    out = [f"# Super-Phase 6 benchmark `{r.digest[:19]}…`\n"]
    out.append(
        f"Manifest `{r.manifest[:19]}…` ({m.name} v{m.version}): {len(r.runs)} runs; memory worlds "
        f"{m.memory_seeds[2]} test + {m.memory_seeds[3]} calibration replicates per environment "
        f"(seed families {m.memory_seeds[0]}, {m.memory_seeds[1]}); belief worlds "
        f"{m.belief_seeds[2]} + {m.belief_seeds[3]} (families {m.belief_seeds[0]}, {m.belief_seeds[1]}); "
        f"jitter {', '.join(f'{n} x[{lo:g},{hi:g}]' for n, lo, hi in m.jitter)}; "
        f"sampling {', '.join(f'{k}={v}' for k, v in m.sampling)}.\n"
    )
    for env, _, _ in ENVIRONMENTS:
        out.append(f"\n## Environment: {env}\n\n### Memory systems\n")
        rows = []
        for s in MEMORY_SYSTEMS:
            c = cells[("memory", env, s)]
            rows.append(
                [
                    s,
                    *(
                        f"{f(c[k][1].mean)} [{f(c[k][1].low)}, {f(c[k][1].high)}]"
                        for k in (
                            "correct_rate",
                            "stale_rate",
                            "none_rate",
                            "evidence_retention",
                            "contested_correct_rate",
                        )
                    ),
                ]
            )
        out.append(
            table(
                [
                    "system",
                    "correct",
                    "stale",
                    "no answer",
                    "evidence retention",
                    "contested correct",
                ],
                rows,
            )
        )
        out.append(f"\n### Belief policies ({env})\n")
        rows = []
        for p in BELIEF_POLICIES:
            c = cells[("belief", env, p)]
            rows.append(
                [
                    p,
                    *(
                        f"{f(c[k][1].mean)} [{f(c[k][1].low)}, {f(c[k][1].high)}]"
                        for k in (
                            "correct_rate",
                            "coverage",
                            "selective_accuracy",
                            "contradiction_correct_rate",
                            "ece_raw",
                            "ece_recal",
                        )
                    ),
                ]
            )
        out.append(
            table(
                [
                    "policy",
                    "correct",
                    "coverage",
                    "selective acc.",
                    "contradiction-heavy correct",
                    "ECE raw",
                    "ECE recalibrated",
                ],
                rows,
            )
        )
    out.append(
        "\n## Paired comparisons against the baseline (per-replicate, Holm within a family)\n"
    )
    for track in ("memory", "belief"):
        for env in (*(e for e, _, _ in ENVIRONMENTS), "ALL"):
            rows = [
                [
                    c.metric,
                    c.treatment,
                    f"{c.diff.mean:+.3f} [{c.diff.low:+.3f}, {c.diff.high:+.3f}]"
                    if c.diff.mean is not None
                    and c.diff.low is not None
                    and c.diff.high is not None
                    else "—",
                    f"{c.diff.positive}/{c.diff.negative}",
                    f"{c.holm_p:.4f}",
                    c.verdict,
                ]
                for c in r.comparisons
                if c.track == track and c.environment == env
            ]
            out.append(f"\n### {track} / {env}\n")
            out.append(
                table(["metric", "treatment", "Δ mean [95% CI]", "+/−", "Holm p", "verdict"], rows)
            )
    out.append("\n## Importance calibration (memory track; fitted on calibration seeds)\n")
    out.append(
        table(
            [
                "environment",
                "system",
                "n fit / test",
                "AUROC use",
                "AUROC truth",
                "ECE raw -> recal",
            ],
            [
                [
                    c.world,
                    c.system,
                    f"{c.n_fit} / {c.n_test}",
                    f(c.auroc_use),
                    f(c.auroc_truth),
                    f"{f(c.raw_ece)} -> {f(c.recalibrated_ece)}",
                ]
                for c in r.calibrations
            ],
        )
    )
    out.append(
        "\n## Provenance, autopsy and replay completeness (pooled over replicates and systems)\n"
    )
    rows = []
    for env, _, _ in ENVIRONMENTS:
        row = [env]
        for track, systems_, keys in (
            (
                "memory",
                MEMORY_SYSTEMS,
                ("lineage_complete", "autopsy_complete", "autopsy_verified", "replay_exact"),
            ),
            ("belief", BELIEF_POLICIES, ("trace_complete", "replay_exact")),
        ):
            for k in keys:
                num = den = 0
                for s in systems_:
                    prop = cells[(track, env, s)][k][0]
                    if prop is not None:
                        num, den = num + prop.numerator, den + prop.denominator
                row.append(f"{num}/{den}")
        rows.append(row)
    out.append(
        table(
            [
                "environment",
                "memory lineage",
                "autopsy complete",
                "autopsy verified",
                "memory replay exact",
                "belief trace complete",
                "belief replay exact",
            ],
            rows,
        )
    )
    out.append("\n## Autopsy demonstration\n")
    out += _autopsy_text(demo.memory_autopsy)
    if demo.replay:
        rp = demo.replay
        out.append(
            f"\nReplay around a source correction: item `{rp.item}` ({rp.key}) at day {rp.recorded_at:g} "
            f"(event `{rp.event}`): just before, {rp.item_before}; just after, {rp.item_after}. "
            f"Snapshots `{rp.before.digest[:19]}…` and `{rp.after.digest[:19]}…`."
        )
    out.append("\nCounterfactuals (evidence held fixed):\n")
    out.append(
        table(
            [
                "change",
                "decisions",
                "changed",
                "why",
                "gained / lost (audit)",
                "re-simulated changed",
                "fixed vs re-sim",
            ],
            [
                [
                    c.label,
                    str(c.decisions),
                    f"{c.changed} ({f(c.changed_share.estimate)})",
                    ", ".join(f"{k}:{v}" for k, v in c.by_why),
                    f"{c.gained} / {c.lost}",
                    str(c.resimulated_changed),
                    str(c.fixed_vs_resimulated),
                ]
                for c in demo.counterfactuals
            ],
        )
    )
    out.append(f"\n{demo.counterfactuals[0].interpretation}.\n")
    out.append("\n" + "\n".join(_autopsy_text(demo.belief_autopsy)))
    return "\n".join(out) + "\n"
