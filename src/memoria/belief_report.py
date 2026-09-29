"""The text report of the belief study: every table in the experiment document is read from
the stored study through this function."""

from __future__ import annotations

from collections.abc import Sequence

from memoria.belief_cases import CaseResult
from memoria.belief_demo import Demonstration
from memoria.belief_run import BeliefStudy
from memoria.calibration import Reliability
from memoria.statistics import PairedMean, Proportion


def _p(x: Proportion | None) -> str:
    if x is None:
        return "—"
    if x.estimate is None:
        return f"{x.numerator}/{x.denominator}"
    return f"{x.numerator}/{x.denominator}={x.estimate:.2f} [{x.low:.2f},{x.high:.2f}]"


def _f(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _pm(m: PairedMean) -> str:
    if m.mean is None:
        return "—"
    ci = f" [{m.low:+.2f},{m.high:+.2f}]" if m.low is not None else ""
    dz = f" dz={m.d_z:+.2f}" if m.d_z is not None else ""
    return f"{m.mean:+.3f}{ci} n={m.n} sign p={m.sign_p:.2g}{dz}"


def _rel(r: Reliability) -> str:
    return " ".join(
        f"[{b.lo:.1f}:{b.n}:{b.accuracy.estimate:.2f}/{b.mean_score:.2f}]"
        for b in r.bins
        if b.n and b.accuracy and b.accuracy.estimate is not None and b.mean_score is not None
    )


def report(study: BeliefStudy, demo: Demonstration) -> str:
    out = ["# Phase 11-12 report", "", f"study {study.digest}", ""]
    mx = study.matrix
    out += ["## Policies, pooled over worlds (test replicates)", "",
            "| policy | n | coverage | correct (of all) | selective acc | ECE raw | ECE recal. | "
            "Brier | overconf. | AUROC | AURC (oracle) | confident wrong |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in mx.pooled:
        s = c.summary
        out.append(
            f"| {c.policy} | {s.n} | {_p(s.coverage)} | {_p(s.correct)} | {_p(s.selective_accuracy)} | "
            f"{_f(s.reliability.ece)} | {_f(s.calibrated.ece if s.calibrated else None)} | "
            f"{_f(s.reliability.brier)} | {_f(s.reliability.overconfidence)} | {_f(s.auroc)} | "
            f"{_f(s.risk.aurc)} ({_f(s.risk.oracle_aurc)}) | {_p(s.confident_wrong)} |"
        )
    names = sorted({c.policy for c in mx.cells})
    worlds = list(dict.fromkeys(c.world for c in mx.cells))
    out += ["", "## Correct answers of all probes, per world", "",
            "| world | " + " | ".join(n.split(":")[0] for n in names) + " |",
            "|---|" + "---|" * len(names)]  # fmt: skip
    by = {(c.world, c.policy): c.summary for c in mx.cells}
    for w in worlds:
        out.append(f"| {w} | " + " | ".join(
            f"{by[(w, n)].correct.numerator}/{by[(w, n)].n} ({by[(w, n)].answered})" for n in names) + " |")  # fmt: skip
    out += ["", "## Paired comparisons with the evidence-count baseline", "",
            "| world | policy | n | correct difference [95% CI], Holm p | key level | Brier (key level) |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for cmp in mx.comparisons:
        d, lo, hi, p, under = cmp.correct
        cell = (
            "—" if d is None else f"{d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.2g}{' u' if under else ''}"
        )
        out.append(f"| {cmp.world} | {cmp.policy} | {cmp.n} | {cell} | {_pm(cmp.keys)} | "
                   f"{_pm(cmp.brier) if cmp.brier else '—'} |")  # fmt: skip
    out += ["", "## Reliability (bin: n : accuracy / mean score)", ""]
    for c in mx.pooled:
        out.append(f"- {c.policy} raw: {_rel(c.summary.reliability)}")
        if c.summary.calibrated:
            out.append(f"- {c.policy} recalibrated: {_rel(c.summary.calibrated)}")
    out += ["", "## Calibration by subgroup (ECE, n answered)", "",
            "| subgroup | " + " | ".join(n.split(":")[0] for n in names) + " |",
            "|---|" + "---|" * len(names)]  # fmt: skip
    sg = {(s.policy, s.tag): s for s in mx.subgroups}
    for tag in dict.fromkeys(s.tag for s in mx.subgroups):
        out.append(f"| {tag} | " + " | ".join(
            f"{_f(sg[(n, tag)].raw.ece, 2)} ({sg[(n, tag)].raw.n})" for n in names) + " |")  # fmt: skip
    out += ["", "## Abstention and uncertainty", "",
            "| policy | abstained | prevented | lost | net | precision |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for qp, q in mx.quality:
        out.append(
            f"| {qp} | {q.abstained} | {q.prevented} | {q.lost} | {q.net} | {_p(q.precision)} |"
        )
    out += ["", "AUROC of each uncertainty component for error (0.5: uninformative):", ""]
    for up, comps in mx.uncertainty:
        out.append(f"- {up}: " + ", ".join(f"{n} {_f(v, 2)}" for n, v in comps))
    out += ["", "## Selective prediction", "", "| policy | coverage | accuracy | AURC | oracle | @50% | @80% |",
            "|---|---|---|---|---|---|---|"]  # fmt: skip
    for sr in study.selective:
        out.append(f"| {sr.policy} | {_f(sr.coverage, 2)} | {_f(sr.accuracy)} | {_f(sr.aurc)} | "
                   f"{_f(sr.oracle_aurc)} | {_f(sr.sel_at_50)} | {_f(sr.sel_at_80)} |")  # fmt: skip
    out += ["", "## Order sensitivity (same evidence, different arrival orders)", "",
            "| policy | orders x worlds | same final beliefs | answer agreement | mid-horizon agreement | "
            "state changes (mean) | correct (min/mean/max) | score range | history mismatches |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for n in names:
        rs = [o for o in study.orders if o.policy == n]
        k = len(rs)
        out.append(
            f"| {n} | {rs[0].orders} x {k} | {sum(o.same_final for o in rs)}/{sum(o.orders for o in rs)} | "
            f"{sum(o.answer_agreement for o in rs) / k:.3f} | {sum(o.mid_agreement for o in rs) / k:.3f} | "
            f"{sum(o.revisions[1] for o in rs) / k:.1f} | {min(o.correct[0] for o in rs):.2f}/"
            f"{sum(o.correct[1] for o in rs) / k:.2f}/{max(o.correct[2] for o in rs):.2f} | "
            f"{sum(o.confidence_range for o in rs) / k:.3f} | {sum(o.historical_mismatches for o in rs)} |"
        )
    out += ["", "Per world (same final beliefs / orders):", ""]
    for w in worlds:
        if w == "ALL":
            continue
        out.append(f"- {w}: " + ", ".join(
            f"{o.policy.split(':')[0]} {o.same_final}/{o.orders}" for o in study.orders if o.world == w))  # fmt: skip
    out += ["", "## Stability (reference arrival order)", "",
            "| policy | revisions/event | churn | oscillations | score volatility | conflict persistence | "
            "days to resolve | latency (days) | unresolved at end |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for n in names:
        ss = [s for s in study.stability if s.policy == n]

        def mean(xs: Sequence[float | None]) -> float | None:
            ys = [x for x in xs if x is not None]
            return sum(ys) / len(ys) if ys else None

        out.append(
            f"| {n} | {_f(mean([s.revisions_per_event for s in ss]))} | {_f(mean([s.churn for s in ss]))} | "
            f"{_f(mean([s.oscillations for s in ss]))} | {_f(mean([s.confidence_volatility for s in ss]))} | "
            f"{_f(mean([s.persistence for s in ss]))} | {_f(mean([s.time_to_resolution_days for s in ss]), 1)} | "
            f"{_f(mean([s.latency_days for s in ss]), 1)} | {sum(s.unresolved_at_end for s in ss)} |"
        )
    out += ["", "## Learned source trust against the hidden reliability", ""]
    for t in study.trust:
        out.append(f"- {t.world}: {t.sources} sources, MAE {_f(t.mae)}, Spearman {_f(t.spearman, 2)}, "
                   f"circular={t.circular}, best {t.best_learned}, worst {t.worst_learned}")  # fmt: skip
    out += ["", "## Contradiction taxonomy (designed cases)", ""]
    out.append(
        f"{sum(tc.ok for tc in study.taxonomy)}/{len(study.taxonomy)} classified as designed"
    )
    for tc in study.taxonomy:
        if not tc.ok:
            out.append(f"- MISS {tc.name}: expected {tc.expected}, got {tc.got}")
    out += ["", "## Contradiction survey of the worlds' claims", "",
            "| world | claims | relations | clusters | open / resolved | genuine relations with an erroneous report | "
            "erroneous concurrent reports flagged | relations between two correct reports read as genuine |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for sv in study.surveys:
        out.append(f"| {sv.world} | {sv.claims} | {dict(sv.relations)} | {sv.clusters} | {sv.open}/{sv.resolved} | "
                   f"{_p(sv.genuine_precision)} | {_p(sv.genuine_recall)} | {_p(sv.changes_misread)} |")  # fmt: skip
    out += ["", "## Adversarial cases (mode value score; ! = overconfident)", "",
            "| case | " + " | ".join(n.split(":")[0] for n in names) + " |",
            "|---|" + "---|" * len(names)]  # fmt: skip
    for case in dict.fromkeys(c.case for c in study.cases):
        rows = {(c.policy, c.lineage, c.identity): c for c in study.cases if c.case == case}
        variants = sorted({(lin, ident) for _, lin, ident in rows})
        for lin, ident in variants:
            label = case + (" +lineage" if lin else "") + (" +identity" if ident else "")
            out.append(f"| {label} | " + " | ".join(
                _case_cell(rows.get((n, lin, ident))) for n in names) + " |")  # fmt: skip
    out += ["", "## Memory operations (pooled over worlds; changes against the raw condition)", "",
            "| condition | policy | correct | selective acc | mean score | ECE raw (d) | ECE recal. (d) | "
            "confident wrong | more confident, not more correct |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for o in study.ops:
        if o.world != "ALL":
            continue
        s = o.summary
        out.append(
            f"| {o.condition} | {o.policy} | {_p(s.correct)} | {_p(s.selective_accuracy)} | "
            f"{_f(o.d_mean_score) if o.d_mean_score is not None else '—'} (d) | "
            f"{_f(s.reliability.ece)} ({_f(o.d_ece)}) | "
            f"{_f(s.calibrated.ece if s.calibrated else None)} ({_f(o.d_ece_calibrated)}) | "
            f"{_p(s.confident_wrong)} | {'YES' if o.more_confident_not_more_correct else ''} |"
        )
    out += ["", "## Belief against retrieval", "",
            "| world | n | retrieval correct | belief correct | both / retr. only / belief only / neither | "
            "McNemar p | disagree | Spearman | ECE retr. / belief | AUROC retr. / belief | AURC retr. / belief |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in study.retrieval:
        out.append(
            f"| {r.world} | {r.n} | {_p(r.retrieval_correct)} | {_p(r.belief_correct)} | {r.both}/{r.retrieval_only}/"
            f"{r.belief_only}/{r.neither} | {r.p_value:.2g} | {_p(r.disagree)} | {_f(r.rank_correlation, 2)} | "
            f"{_f(r.retrieval_ece, 2)} / {_f(r.belief_ece, 2)} | {_f(r.retrieval_auroc, 2)} / "
            f"{_f(r.belief_auroc, 2)} | {_f(r.retrieval_aurc, 2)} / {_f(r.belief_aurc, 2)} |"
        )
    out += ["", "## Demonstration", "", demo.model_dump_json(indent=1)[:24000]]
    return "\n".join(out)


def _case_cell(c: CaseResult | None) -> str:
    if c is None:
        return ""
    v = c.value or c.mode.replace("_", "-")
    s = "" if c.score is None else f" {c.score:.2f}"
    return f"{v}{s}{'!' if c.overconfident else ''}"
