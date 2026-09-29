"""Belief calibration after memory operations, and belief against retrieval.

Memory operations change the evidence a belief system sees: interference adds distractor
reports, forgetting withdraws evidence, consolidation adds (or replaces raw evidence with)
derived reports. Each is built from the existing Phase 7-10 machinery (an episodic log, a
:class:`~memoria.consolidation.Hierarchy`, a :class:`~memoria.forgetting.ForgettingRecord`,
an injected dataset), then the same belief policy runs over the changed evidence. The
question is whether confidence stays calibrated, and in particular whether the system
becomes more confident without becoming more correct.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from memoria.artifacts import ArtifactStore
from memoria.belief_lab import Summary, predictions, run_world, summarize, with_signals
from memoria.belief_study import _brier_keys
from memoria.belief_worlds import BeliefWorld
from memoria.beliefs import POLICIES, BeliefPolicy, EvidenceItem, evidence_from_statement
from memoria.calibration import (
    Isotonic,
    Prediction,
    Reliability,
    answered_pairs,
    auroc,
    reliability,
    risk_coverage,
)
from memoria.consolidation import consolidate
from memoria.consolidation_eval import RETRIEVAL
from memoria.core import EpistemicStatus, Level, Record, quantize
from memoria.embeddings import HashedNgramEmbedder
from memoria.forgetting import apply, forget
from memoria.formation import EpisodicPolicy, form
from memoria.hybrid import Corpus, Engine, HybridQuery
from memoria.memory_lab import FORGETTING, HIERARCHICAL, world_distractors
from memoria.revision import RunMeta
from memoria.scenarios import day
from memoria.sources import spearman
from memoria.statistics import (
    PairedMean,
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    significant,
)
from memoria.store import MemoryLog

CONDITIONS = ("raw", "interference", "forgetting", "cons-add", "cons-replace", "cons-add-forget")
OPS_POLICIES = ("A:evidence-count", "B:source-weighted", "E:corroboration", "H:conservative")


@dataclass
class Condition:
    name: str
    evidence: list[EvidenceItem]
    withdrawals: list[tuple[datetime, str]] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    forgetting: str = "none"
    consolidation: str = "none"
    interference: str = "none"
    derived: bool = False


def memory_log(world: BeliefWorld) -> MemoryLog:
    log = MemoryLog(":memory:")
    for s in world.dataset.steps:
        form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
    return log


def _forget_at(log: MemoryLog, world: BeliefWorld, fraction: float, store: ArtifactStore | None
               ) -> tuple[list[tuple[datetime, str]], str]:  # fmt: skip
    t = day(fraction * world.spec.horizon_days)
    corpus = Corpus.from_log(log, t)
    record = forget(corpus, FORGETTING["validity"], t)
    apply(corpus, record)  # the corpus identity now names the record (cascade over derived)
    if store is not None:
        store.put_record(record)
    have = {i.id for i in world.evidence}
    hidden = record.unavailable()
    out = [
        (t, x.digest)
        for e in corpus.entries
        if e.level is Level.L1 and e.digest in hidden
        for x in e.sources
        if x.digest in have
    ]
    return sorted(out), record.digest


def _derived_items(log: MemoryLog, world: BeliefWorld, fraction: float, store: ArtifactStore | None
                   ) -> tuple[list[EvidenceItem], set[str], str]:  # fmt: skip
    t = day(fraction * world.spec.horizon_days)
    corpus = Corpus.from_log(log, t)
    h = consolidate(corpus, HIERARCHICAL, t)
    if store is not None:
        store.put_record(h)
    source = {s.experience.digest: s.experience.source for s in world.dataset.steps}
    items: list[EvidenceItem] = []
    covered: set[str] = set()
    for m in h.memories:
        if m.level is not Level.L2 or m.key is None or m.value is None:
            continue
        covered.update(m.derived_from)
        items.append(EvidenceItem(
            id=m.digest, key=m.key, verb="set", value=m.value, raw=m.value,
            occurred_at=m.valid_from, recorded_at=t, source=f"derived:{m.memory_id}",
            status=EpistemicStatus.DERIVED, multiplicity=len(m.derived_from),
            lineage=tuple(sorted((e, source[e]) for e in m.derived_from)),
        ))  # fmt: skip
    return items, covered, h.digest


def build_condition(world: BeliefWorld, name: str, store: ArtifactStore | None = None) -> Condition:
    raw = list(world.evidence)
    if name == "raw":
        return Condition(name, raw)
    log = memory_log(world)
    with log:
        if name == "interference":
            steps = world_distractors(world.dataset).steps
            extra = [
                it
                for s in steps
                if (it := evidence_from_statement(s.experience.digest, s.experience.content,
                                                  s.experience.occurred_at, s.recorded_at,
                                                  s.experience.source)) is not None
            ]  # fmt: skip
            ident = "sha256:" + "0" * 64
            return Condition(name, raw + extra, tags=("interference-env",), interference=ident)
        if name == "forgetting":
            w, rec = _forget_at(log, world, 0.6, store)
            return Condition(name, raw, w, ("forgotten-env",), forgetting=rec)
        derived, covered, h = _derived_items(log, world, 0.5, store)
        t = day(0.5 * world.spec.horizon_days)
        have = {i.id for i in raw}
        if name == "cons-add":
            return Condition(name, raw + derived, tags=("consolidated",), consolidation=h,
                             derived=True)  # fmt: skip
        if name == "cons-replace":
            w = sorted((t, e) for e in covered if e in have)
            return Condition(name, raw + derived, w, ("consolidated",), consolidation=h,
                             derived=True)  # fmt: skip
        if name == "cons-add-forget":
            w, rec = _forget_at(log, world, 0.8, store)
            return Condition(name, raw + derived, w, ("consolidated", "forgotten-env"),
                             forgetting=rec, consolidation=h, derived=True)  # fmt: skip
    raise ValueError(f"unknown condition {name!r}")


class OpsCell(Record):
    world: str  # "ALL" for pooled
    condition: str
    policy: str  # a policy name, "+lineage" if lineage-aware
    summary: Summary
    d_ece: float | None  # against the raw condition, same policy, raw score
    d_ece_calibrated: float | None  # recalibration fitted on raw calibration replicates
    d_correct: float | None
    d_mean_score: float | None  # mean score of answered probes
    d_selective_accuracy: float | None
    more_confident_not_more_correct: bool
    # Pooled cells only: correct answers against the raw condition on identical probes
    # (difference, low, high, Holm p across conditions of one policy, underpowered), and the
    # key-level paired difference in squared error over probes both conditions answered.
    paired_correct: tuple[float | None, float | None, float | None, float | None, bool] | None = (
        None
    )
    brier_keys: PairedMean | None = None


def _mean_score(preds: Sequence[Prediction]) -> float | None:
    xs = [p.score for p in preds if p.answered and p.score is not None]
    return math.fsum(xs) / len(xs) if xs else None


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else quantize(a - b)


def ops_study(
    worlds_: Sequence[BeliefWorld], calibrators: dict[str, Isotonic],
    store: ArtifactStore | None = None, policies: Sequence[str] = OPS_POLICIES,
    conditions: Sequence[str] = CONDITIONS,
) -> tuple[OpsCell, ...]:  # fmt: skip
    """Run every condition and policy over ``worlds_`` (test replicates) and compare each
    with the raw condition of the same policy. Pooled cells have world ``ALL``."""
    preds: dict[tuple[str, str, str], list[Prediction]] = {}
    keys: dict[tuple[str, str, str], list[str]] = {}
    variants: dict[tuple[str, bool], BeliefPolicy] = {}
    for name in policies:
        variants[(name, False)] = POLICIES[name]
        variants[(name, True)] = with_signals(POLICIES[name], "lineage",
                                              name=f"{name}+lineage")  # fmt: skip
    for w in worlds_:
        for cname in conditions:
            cond = build_condition(w, cname, store)
            for (pname, lineage), pol in variants.items():
                if lineage and not cond.derived:
                    continue
                meta = RunMeta(world=w.dataset.digest, seed=w.spec.seed,
                               forgetting=cond.forgetting, consolidation=cond.consolidation,
                               interference=cond.interference)  # fmt: skip
                run = run_world(w, pol, evidence=cond.evidence, withdrawals=cond.withdrawals,
                                meta=meta)  # fmt: skip
                preds.setdefault((w.spec.name, cname, pol.name), []).extend(
                    predictions(run, w, calibrator=calibrators[pname], tags=cond.tags)
                )
                keys.setdefault((w.spec.name, cname, pol.name), []).extend(
                    f"{w.spec.seed}:{p.key}" for p in w.probes
                )
    pooled: dict[tuple[str, str], list[Prediction]] = {}
    for (_, cname, pname), ps in preds.items():
        pooled.setdefault((cname, pname), []).extend(ps)

    def cell(
        world: str, cname: str, pname: str, ps: list[Prediction], base: list[Prediction]
    ) -> OpsCell:
        s, b = summarize(ps), summarize(base)
        d_conf = _delta(_mean_score(ps), _mean_score(base))
        d_sel = _delta(
            s.selective_accuracy.estimate if s.selective_accuracy else None,
            b.selective_accuracy.estimate if b.selective_accuracy else None,
        )
        d_corr = _delta(s.correct.estimate, b.correct.estimate)
        flag = bool(d_conf is not None and d_conf > 0.02 and d_sel is not None and d_sel <= 0.0)
        return OpsCell(
            world=world, condition=cname, policy=pname, summary=s,
            d_ece=_delta(s.reliability.ece, b.reliability.ece),
            d_ece_calibrated=_delta(s.calibrated.ece if s.calibrated else None,
                                    b.calibrated.ece if b.calibrated else None),
            d_correct=d_corr, d_mean_score=d_conf, d_selective_accuracy=d_sel,
            more_confident_not_more_correct=flag,
        )  # fmt: skip

    out: list[OpsCell] = []
    base_name = {v.name: k[0] for k, v in variants.items()}
    for (wn, cname, pname), ps in sorted(preds.items()):
        base = preds[(wn, "raw", base_name[pname])]
        out.append(cell(wn, cname, pname, ps, base))
    pooled_keys: dict[tuple[str, str], list[str]] = {}
    for (wn, cname, pname), ks in keys.items():
        pooled_keys.setdefault((cname, pname), []).extend(f"{wn}/{k}" for k in ks)
    rows = []
    for (cname, pname), ps in sorted(pooled.items()):
        base = pooled[("raw", base_name[pname])]
        c = cell("ALL", cname, pname, ps, base)
        a, b = (
            [bool(p.answered and p.correct) for p in base],
            [bool(p.answered and p.correct) for p in ps],
        )
        both = sum(x and y for x, y in zip(a, b, strict=True))
        f = sum(x and not y for x, y in zip(a, b, strict=True))
        g = sum(y and not x for x, y in zip(a, b, strict=True))
        diff = newcombe_paired(both, g, f, len(a) - both - f - g, 0.95)
        rows.append(
            (c, pname, cname, diff, significant(mcnemar_exact(f, g)),
             min_achievable_p(f + g) > 0.05, base, ps, pooled_keys[(cname, pname)])
        )  # fmt: skip
    adjusted: dict[tuple[str, str], float] = {}
    for pname in {r[1] for r in rows}:
        family = [r for r in rows if r[1] == pname and r[2] != "raw"]
        for r, q in zip(family, holm([r[4] for r in family]), strict=True):
            adjusted[(r[2], pname)] = significant(q)
    for c, pname, cname, diff, _, under, base, ps, ks in rows:
        paired = None
        if cname != "raw" and diff is not None:
            holm_p = adjusted[(cname, pname)]
            paired = (quantize(diff[0]), quantize(diff[1]), quantize(diff[2]), holm_p, under)
        brier = _brier_keys(base, ps, ks, 0.95)
        out.append(c.model_copy(update={"paired_correct": paired, "brier_keys": brier}))
    return tuple(out)


# --- belief against retrieval --------------------------------------------------------------------


class RetrievalComparison(Record):
    world: str
    n: int
    retrieval_correct: Proportion  # top-1 retrieved memory states the true value
    belief_correct: Proportion  # answered by belief and right (unanswered = not right)
    both: int
    retrieval_only: int
    belief_only: int
    neither: int
    p_value: float  # exact McNemar on the discordant probes
    disagree: Proportion  # top-1 retrieved value differs from the belief's answer
    rank_correlation: float | None  # Spearman: retrieval final score vs belief score
    retrieval_ece: float | None  # the relevance score read as a probability
    belief_ece: float | None
    retrieval_auroc: float | None  # does the score rank right above wrong?
    belief_auroc: float | None
    retrieval_aurc: float | None
    belief_aurc: float | None
    retrieval_reliability: Reliability


def retrieval_vs_belief(world: BeliefWorld, preds: Sequence[Prediction],
                        confidence: float = 0.95) -> RetrievalComparison:  # fmt: skip
    """Rank probes by the retrieval score and by the belief score: relevance is not
    support. Retrieval is the Phase 7 mixed policy (attribute, lexical and semantic
    signals, temporal filter) over the episodic memories known at each probe's ``known_at``."""
    embedder = HashedNgramEmbedder()
    probes = {p.id: p for p in world.dataset.probes}
    truth = {p.id: p.truth for p in world.probes}
    rows: list[tuple[float, bool, Prediction]] = []
    with memory_log(world) as log:
        for pr in preds:
            probe = probes[pr.probe]
            corpus = Corpus.from_log(log, probe.known_at)
            q = HybridQuery(text=probe.text, valid_at=probe.valid_at, known_at=probe.known_at,
                            limit=1, key=probe.key)  # fmt: skip
            trace = Engine(corpus, embedder).retrieve(RETRIEVAL["mixed"], q)
            by = {e.digest: e for e in corpus.entries}
            top = by[trace.selected[0]] if trace.selected else None
            value = (
                " ".join(top.claim.value.split()).casefold()
                if top is not None and top.claim is not None and top.claim.value is not None
                else None
            )
            score = trace.ranking[0].final if trace.ranking else 0.0
            rows.append((score, value == truth[pr.probe] if value is not None else False, pr))
            rows[-1] = (score, rows[-1][1], pr.model_copy(update={"value": value}))
    n = len(rows)
    ret_ok = [r[1] for r in rows]
    bel_ok = [bool(p.answered and p.correct) for _, _, p in rows]
    both = sum(a and b for a, b in zip(ret_ok, bel_ok, strict=True))
    r_only = sum(a and not b for a, b in zip(ret_ok, bel_ok, strict=True))
    b_only = sum(b and not a for a, b in zip(ret_ok, bel_ok, strict=True))
    norm = [max(s, 0.0) for s, _, _ in rows]
    top_norm = max(norm) or 1.0
    ret_pairs = [(x / top_norm, ok) for x, ok in zip(norm, ret_ok, strict=True)]
    orig = {p.probe: p for p in preds}
    pairs_b = answered_pairs(preds)
    both_scored = [(s, sc) for s, _, p in rows if (sc := orig[p.probe].score) is not None]
    disagree = sum(
        1
        for _, _, p in rows
        if orig[p.probe].answered and p.value is not None and p.value != orig[p.probe].value
    )
    return RetrievalComparison(
        world=world.spec.name, n=n, retrieval_correct=Proportion.of(sum(ret_ok), n, confidence),
        belief_correct=Proportion.of(sum(bel_ok), n, confidence), both=both, retrieval_only=r_only,
        belief_only=b_only, neither=n - both - r_only - b_only,
        p_value=significant(mcnemar_exact(r_only, b_only)),
        disagree=Proportion.of(disagree, n, confidence),
        rank_correlation=spearman([a for a, _ in both_scored], [b for _, b in both_scored]),
        retrieval_ece=reliability(ret_pairs).ece, belief_ece=reliability(pairs_b).ece,
        retrieval_auroc=auroc(ret_pairs), belief_auroc=auroc(pairs_b),
        retrieval_aurc=risk_coverage(ret_pairs).aurc, belief_aurc=risk_coverage(pairs_b).aurc,
        retrieval_reliability=reliability(ret_pairs),
    )  # fmt: skip
