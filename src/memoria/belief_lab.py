"""The belief-revision and calibration laboratory (Phases 11-12).

Research questions (docs/experiments/phase11-12-beliefs-calibration.md):

- RQ-B1..B4: how to represent incompatible claims, revise beliefs, tell contradiction from
  change, uncertainty, source disagreement and correction, and whether confidence can be
  calibrated;
- RQ-U1..U3: whether confidence is a probability or a ranking signal, what drives its
  calibration, and what forgetting, consolidation and interference do to it.

Every result is computed from a :class:`~memoria.revision.LedgerRun` (the replayable
ledger) and the hidden ground truth of a generated world, on test seeds disjoint from the
calibration seeds on which any recalibration is fitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from memoria.belief_worlds import BeliefProbe, BeliefWorld
from memoria.beliefs import BeliefPolicy, EvidenceItem, SignalUse
from memoria.calibration import (
    Isotonic,
    Prediction,
    Reliability,
    RiskCoverage,
    answered_pairs,
    auroc,
    reliability,
    risk_coverage,
)
from memoria.core import Record, quantize
from memoria.entities import CONSERVATIVE, resolve
from memoria.revision import (
    LedgerRun,
    RunMeta,
    SelectivePolicy,
    covering,
    decide,
    run_ledger,
)
from memoria.scenarios import day
from memoria.statistics import Proportion

SELECTIVE = SelectivePolicy(name="answer-0.6-hedge-0.35", answer_at=0.6, uncertain_at=0.35)
DIGEST0 = "sha256:" + "0" * 64


def identity_map(keys: Sequence[str]) -> dict[str, float]:
    """Identity uncertainty per entity: the highest similarity of an unmerged resolution
    candidate (Phase 8 entity resolution over the entities the keys name)."""
    entities = sorted({k.split(".", 1)[0] for k in keys})
    r = resolve({e: [DIGEST0] for e in entities}, {}, CONSERVATIVE)
    out: dict[str, float] = {}
    for d in r.decisions:
        if d.rule == "similar" and not d.accepted:
            for m in (d.a, d.b):
                ent = m.removeprefix("structured:")
                out[ent] = max(out.get(ent, 0.0), d.similarity)
    return out


def with_signals(policy: BeliefPolicy, *extra: str, name: str | None = None) -> BeliefPolicy:
    """The policy with more signals (for example ``lineage`` or ``identity``)."""
    have = {s.name for s in policy.signals}
    signals = sorted(
        [*policy.signals, *(SignalUse(name=n) for n in extra if n not in have)],  # type: ignore[arg-type]
        key=lambda s: s.name,
    )
    return policy.model_copy(update={"signals": tuple(signals), "name": name or policy.name})


def run_world(
    world: BeliefWorld,
    policy: BeliefPolicy,
    *,
    evidence: Sequence[EvidenceItem] | None = None,
    withdrawals: Sequence[tuple[datetime, str]] = (),
    meta: RunMeta | None = None,
) -> LedgerRun:
    m = meta or RunMeta(world=world.dataset.digest, seed=world.spec.seed)
    ident = identity_map(sorted(world.truth)) if policy.signal("identity") else {}
    return run_ledger(
        list(evidence if evidence is not None else world.evidence), policy, world.sources,
        identity=ident, withdrawals=withdrawals, meta=m,
    )  # fmt: skip


# --- predictions and summaries -------------------------------------------------------------------


def key_tags(world: BeliefWorld) -> dict[str, tuple[str, ...]]:
    """Static subgroup tags of each key, from the world (never from outcomes)."""
    by_key: dict[str, list[EvidenceItem]] = {}
    for it in world.evidence:
        by_key.setdefault(it.key, []).append(it)
    out = {}
    for key, items in by_key.items():
        tags = []
        ent, attr = key.split(".", 1)
        if attr == "dose":
            tags.append("numeric")
        if world.spec.near_collisions and ent in ("ana", "anna", "ben", "benn"):
            tags.append("entity-collision")
        if len(world.truth[key]) >= 3:
            tags.append("temporal")
        clusters = 0
        last = None
        seen: set[str] = set()
        for it in sorted(items, key=lambda i: i.occurred_at):
            if last is not None and (it.occurred_at - last).days < 1:
                seen.add(it.value or "")
                if len(seen) > 1:
                    clusters += 1
                    seen = set()
            else:
                seen = {it.value or ""}
            last = it.occurred_at
        if clusters >= 2:
            tags.append("contradiction-heavy")
        out[key] = tuple(tags)
    return out


def predictions(
    run: LedgerRun,
    world: BeliefWorld,
    *,
    sel: SelectivePolicy = SELECTIVE,
    calibrator: Isotonic | None = None,
    tags: Sequence[str] = (),
    probes: Sequence[BeliefProbe] | None = None,
) -> list[Prediction]:
    ktags = key_tags(world)
    out: list[Prediction] = []
    for pr in probes if probes is not None else world.probes:
        valid, known = day(pr.valid_at_day), day(pr.known_at_day)
        kb = run.at(pr.key, known)
        cover = covering(kb, valid)
        cal = {b.id: calibrator(b.score) for b in cover} if calibrator else None
        d = decide(pr.key, kb, valid, known, sel)
        best = max(cover, key=lambda b: (b.score, b.id), default=None)
        chosen = next((b for b in cover if b.id in d.beliefs), None) if d.beliefs else None
        answered = d.mode in ("answer", "answer_with_uncertainty")
        frac = pr.known_at_day / world.spec.horizon_days
        btags: list[str] = [
            "time:early" if frac < 0.34 else "time:mid" if frac < 0.67 else "time:late"
        ]
        if chosen is not None:
            btags.append(
                "evidence:"
                + ("3+" if chosen.independent_count >= 3 else str(chosen.independent_count))
            )
            depth = max((world.sources.depth(s) for s in chosen.sources), default=1)
            btags.append("depth:" + ("2+" if depth >= 2 else "1"))
            btags += sorted({"src:" + s.split(":")[0] for s in chosen.sources})
        out.append(Prediction(
            probe=pr.id, key=pr.key, mode=d.mode,
            value=d.values[0] if answered and d.values else None,
            score=d.score if chosen is not None else None,
            calibrated=quantize(cal[chosen.id]) if cal and chosen is not None else None,
            correct=(d.values[0] == pr.truth) if answered and d.values else None,
            would_be_correct=(best.value == pr.truth) if best is not None else None,
            candidate_hit=(pr.truth in d.values) if d.mode == "competing" else None,
            tags=tuple(sorted({*ktags.get(pr.key, ()), *btags, *tags})),
            uncertainty=tuple(sorted(chosen.uncertainty.model_dump().items())) if chosen else (),
        ))  # fmt: skip
    return out


class Summary(Record):
    """One prediction set, summarised. Every count is exact; nothing is imputed."""

    n: int
    answered: int
    coverage: Proportion
    correct: Proportion  # correct answers of all probes (not answering counts as not correct)
    selective_accuracy: Proportion | None  # correct answers of answered probes
    modes: tuple[tuple[str, int], ...]
    competing_hit: Proportion | None  # competing: truth among the presented values
    reliability: Reliability  # of the raw score over answered probes
    calibrated: Reliability | None  # of the recalibrated value, if any
    confident_wrong: Proportion | None  # answered with score >= 0.8 and wrong
    risk: RiskCoverage
    auroc: float | None  # does the score rank right above wrong?


def summarize(preds: Sequence[Prediction], confidence: float = 0.95) -> Summary:
    n = len(preds)
    ans = [p for p in preds if p.answered and p.correct is not None]
    right = sum(1 for p in ans if p.correct)
    modes: dict[str, int] = {}
    for p in preds:
        modes[p.mode] = modes.get(p.mode, 0) + 1
    pairs = answered_pairs(preds)
    cal_pairs = answered_pairs(preds, calibrated=True)
    comp = [p for p in preds if p.mode == "competing"]
    conf = [p for p in ans if (p.score or 0) >= 0.8]
    return Summary(
        n=n, answered=len(ans), coverage=Proportion.of(len(ans), n, confidence),
        correct=Proportion.of(right, n, confidence),
        selective_accuracy=Proportion.of(right, len(ans), confidence) if ans else None,
        modes=tuple(sorted(modes.items())),
        competing_hit=Proportion.of(sum(1 for p in comp if p.candidate_hit), len(comp), confidence)
        if comp else None,
        reliability=reliability(pairs),
        calibrated=reliability(cal_pairs) if cal_pairs else None,
        confident_wrong=Proportion.of(sum(1 for p in conf if not p.correct), len(conf), confidence)
        if conf else None,
        risk=risk_coverage(pairs), auroc=auroc(pairs),
    )  # fmt: skip
