"""Markdown report for the Super-Phase 5 study. Every number is read from the stored study."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from memoria.adaptive_study import (
    ADAPTIVE,
    AdaptiveStudy,
    Interval,
    Summary,
    systems,
)
from memoria.adaptive_worlds import CHECKPOINTS
from memoria.statistics import PairedMean, Proportion

MAIN = (
    "A:static-latest",
    "B:adaptive-rank",
    "C:age-30",
    "D:age-90",
    "E:gate-latest",
    "F:gate-adaptive",
    "G:gate-stale-aware",
    "H:static-importance",
    "I:gold-extraction",
    "J:similarity-identity",
    "K:feedback-lag-7",
)


def f(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def prop(p: Proportion | None) -> str:
    if p is None or p.estimate is None or p.low is None or p.high is None:
        return "—"
    return f"{p.estimate:.3f} [{p.low:.3f}, {p.high:.3f}]"


def ival(i: Interval) -> str:
    if i.mean is None:
        return "—"
    if i.low is None or i.high is None:
        return f"{i.mean:.3f}"
    return f"{i.mean:.3f} [{i.low:.3f}, {i.high:.3f}]"


def diff(pm: PairedMean) -> str:
    if pm.mean is None:
        return "—"
    if pm.low is None or pm.high is None:
        return f"{pm.mean:+.3f}"
    return f"{pm.mean:+.3f} [{pm.low:+.3f}, {pm.high:+.3f}]"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _metric(s: Summary, name: str) -> tuple[Proportion | None, Interval]:
    return next((p, i) for n, p, i in s.metrics if n == name)


def report(study: AdaptiveStudy) -> str:
    worlds = sorted({s.world for s in study.summaries})
    by = {(s.world, s.system): s for s in study.summaries}
    curves = {(c.world, c.system, c.metric): c for c in study.curves}
    defs = systems()
    out: list[str] = [f"# Super-Phase 5 study `{study.digest[:19]}…`\n"]
    out.append(
        f"{len(study.runs)} runs; worlds: {', '.join(worlds)}. Replicates are independent seeds; "
        "intervals are Wilson (pooled probes) or Student-t (across replicates); paired tests are "
        "per-replicate with Holm within a world × metric family.\n"
    )
    out.append("## Systems\n")
    out.append(
        table(
            [
                "system",
                "ingest",
                "identity",
                "schedule",
                "λ (rank weight)",
                "importance",
                "feedback lag",
            ],
            [
                [
                    n,
                    s.ingest,
                    s.identity,
                    s.schedule.name,
                    f"{s.rank_weight:g}",
                    s.importance.name,
                    f"{s.feedback_lag_days:g}",
                ]
                for n, s in sorted(defs.items())
            ],
        )
    )
    for w in worlds:
        out.append(f"\n## World: {w}\n")
        rows = []
        for n in sorted(defs):
            s = by[(w, n)]
            st, _ = _metric(s, "stale_rate")
            co, _ = _metric(s, "correct_rate")
            no, _ = _metric(s, "none_rate")
            ev, _ = _metric(s, "evidence_retention")
            it, _ = _metric(s, "item_retention")
            rows.append(
                [
                    n,
                    prop(co),
                    prop(st),
                    prop(no),
                    prop(ev),
                    prop(it),
                    ival(s.latency_days),
                    prop(s.latency_censored),
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
                    "item retention",
                    "revision latency (days)",
                    "censored changes",
                ],
                rows,
            )
        )
        out.append(f"\n### Retention curves, stale-answer rate over time ({w})\n")
        out.append(
            table(
                ["system", *(f"{h:g}" for h in CHECKPOINTS)],
                [
                    [
                        n,
                        *(
                            ival(p.across) if p.across.mean is not None else "—"
                            for p in curves[(w, n, "stale_rate")].points
                        ),
                    ]
                    for n in MAIN
                ],
            )
        )
        out.append(f"\n### Evidence retention over time ({w})\n")
        out.append(
            table(
                ["system", *(f"{h:g}" for h in CHECKPOINTS)],
                [
                    [n, *(f(p.across.mean) for p in curves[(w, n, "evidence_retention")].points)]
                    for n in MAIN
                ],
            )
        )
    out.append("\n## Paired comparisons (treatment − baseline, per-replicate)\n")
    for w in worlds:
        rows = []
        for c in study.comparisons:
            if c.world == w and c.metric in ("stale_rate", "correct_rate"):
                rows.append(
                    [
                        c.metric,
                        c.treatment,
                        c.baseline,
                        diff(c.diff),
                        f"{c.diff.positive}/{c.diff.negative}",
                        f"{c.holm_p:.4f}",
                        c.verdict,
                    ]
                )
        out.append(f"\n### {w}\n")
        out.append(
            table(
                ["metric", "treatment", "baseline", "Δ mean [95% CI]", "+/−", "Holm p", "verdict"],
                rows,
            )
        )
    out.append("\n## Onset of stale-memory damage over time\n")
    for w in worlds:
        out.append(f"\n### {w}\n")
        rows = []
        for o in study.onsets:
            if o.world == w:
                rows.append(
                    [
                        o.treatment,
                        o.baseline,
                        *(f"{d.mean:+.3f}" if d.mean is not None else "—" for _, d, _ in o.points),
                        f(o.onset, 0),
                    ]
                )
        out.append(
            table(["treatment", "baseline", *(f"{h:g}" for h in CHECKPOINTS), "onset (day)"], rows)
        )
    out.append("\n## Dose–response: where damage begins\n")
    for d in study.dose_onsets:
        out.append(
            f"\n### {d.world}: {d.family} (baseline {d.baseline}); onset = {f(d.onset, 2)}\n"
        )
        out.append(
            table(
                ["dose", "system", "Δ stale [95% CI]", "Holm p", "item retention"],
                [[f"{x:g}", s, diff(pm), f"{a:.4f}", f(r)] for x, s, pm, a, r in d.doses],
            )
        )
    out.append(
        "\n## Importance ablations (stale-rate and correct-rate change vs the full model, "
        f"system {ADAPTIVE})\n"
    )
    for w in worlds:
        out.append(f"\n### {w}\n")
        out.append(
            table(
                ["removed component", "Δ stale [95% CI]", "Holm p", "Δ correct [95% CI]", "Holm p"],
                [
                    [
                        a.component,
                        diff(a.stale.diff),
                        f"{a.stale.holm_p:.4f}",
                        diff(a.correct.diff),
                        f"{a.correct.holm_p:.4f}",
                    ]
                    for a in study.ablations
                    if a.world == w
                ],
            )
        )
    out.append(
        "\n## Importance calibration (fitted on calibration replicates, evaluated on test)\n"
    )
    out.append(
        table(
            [
                "world",
                "system",
                "n fit / test",
                "P(used again)",
                "AUROC use",
                "AUROC truth",
                "ECE raw → recal",
                "Brier raw → recal",
            ],
            [
                [
                    c.world,
                    c.system,
                    f"{c.n_fit} / {c.n_test}",
                    f(c.base_rate_use),
                    f(c.auroc_use),
                    f(c.auroc_truth),
                    f"{f(c.raw_ece)} → {f(c.recalibrated_ece)}",
                    f"{f(c.raw_brier)} → {f(c.recalibrated_brier)}",
                ]
                for c in study.calibrations
            ],
        )
    )
    out.append("\n## Trace audit (decisions and importance records re-derived from the ledger)\n")
    out.append(
        table(
            [
                "world",
                "system",
                "decisions",
                "importance records",
                "events",
                "cutoff viol.",
                "re-derivation viol.",
                "broken links",
                "archived",
            ],
            [
                [
                    a.world,
                    a.system,
                    str(a.decisions),
                    str(a.importances),
                    str(a.events),
                    str(a.cutoff_violations),
                    str(a.reproduction_violations),
                    str(a.unresolved_importance_links),
                    str(a.archived),
                ]
                for a in study.audits
            ],
        )
    )
    out.append("\n## Free-text extraction\n")
    out.append(
        table(
            [
                "world",
                "identity",
                "reports",
                "precision",
                "recall",
                "uncertain → fact",
                "unresolved reports",
                "claim lineage verified",
                "unresolved reasons",
            ],
            [
                [
                    e.world,
                    e.policy,
                    str(e.reports),
                    prop(e.precision),
                    prop(e.recall),
                    prop(e.conversion),
                    f"{e.unresolved_reports}/{e.reports}",
                    prop(e.claims_verified),
                    ", ".join(f"{k}:{v}" for k, v in e.reasons) or "—",
                ]
                for e in study.extraction
            ],
        )
    )
    out.append("\n## Entity identity (same mentions, three policies)\n")
    out.append(
        table(
            [
                "world",
                "policy",
                "mentions",
                "correct",
                "false merges",
                "false-merge rate",
                "ambiguous",
                "candidates",
                "unknown",
            ],
            [
                [
                    i.world,
                    i.policy,
                    str(i.mentions),
                    str(i.correct),
                    str(i.false_merges),
                    prop(i.false_merge_rate),
                    str(i.ambiguous),
                    str(i.candidates),
                    str(i.unknown),
                ]
                for i in study.identity
            ],
        )
    )
    out.append("\n## Systems' cost (full horizon)\n")
    for w in worlds:
        out.append(f"\n### {w}\n")
        out.append(
            table(
                [
                    "system",
                    "retrievable items / probe",
                    "items scanned / probe",
                    "events",
                    "importance evals",
                    "misattributed claims",
                    "unresolved / (unresolved + items)",
                    "users told correct",
                ],
                [
                    [
                        n,
                        f(by[(w, n)].items_per_probe_available, 2),
                        f(by[(w, n)].scanned_per_probe, 2),
                        str(by[(w, n)].events),
                        str(by[(w, n)].importance_evals),
                        prop(by[(w, n)].misattributed),
                        prop(by[(w, n)].unresolved),
                        prop(by[(w, n)].user_correct),
                    ]
                    for n in MAIN
                ],
            )
        )
    out.append(
        "\n## Provenance completeness (answers whose item resolves to its text and claim span)\n"
    )
    out.append(
        table(
            ["world", *(n for n in MAIN[:8])],
            [
                [w, *(prop(_metric(by[(w, n)], "lineage_complete")[0]) for n in MAIN[:8])]
                for w in worlds
            ],
        )
    )
    out.append("\n## Designed cases\n")
    fails = [k for k in study.cases if not k.passed]
    out.append(f"{len(study.cases) - len(fails)}/{len(study.cases)} passed.")
    for k in study.cases:
        if k.family == "merge" or not k.passed or "FALSE" in k.got:
            out.append(
                f"- {k.family}/{k.name} [{k.policy}]: {'ok' if k.passed else 'FAILED'} - {k.got}"
            )
    return "\n".join(out) + "\n"
