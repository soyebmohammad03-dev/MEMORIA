"""One end-to-end demonstration of the MEMORIA lifecycle.

Run it with ``python -m memoria.lifecycle_demo <dir>``.

It composes two existing, fixed demonstrations and adds nothing new to the science:

* the memory-substrate lifecycle (``memory_lab.demonstration``, generated "contradictory" world):
  experiences → formation → consolidation → graph → hybrid retrieval → contradiction exposure →
  targeted forgetting → interference injection → provenance autopsy;
* the adaptive lifecycle (``autopsy_demo.demonstrate``): feedback-driven importance and retention →
  belief revision → temporal replay around a source correction → counterfactual replay →
  memory and belief autopsy.

Both are then verified: the store re-hashes every object, and both demonstrations are rebuilt into
an independent store and must reproduce the same digests. Nothing here is a benchmark result:
it is one world, one path through the pipeline, chosen by fixed rules, to show the artifacts.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from memoria.artifacts import ArtifactStore
from memoria.autopsy import Autopsy
from memoria.autopsy_demo import AutopsyDemonstration, demonstrate
from memoria.core import Digest, RepresentationSpec
from memoria.embeddings import HashedNgramEmbedder
from memoria.memory_lab import Demonstration, demonstration
from memoria.statistics import Proportion


def _p(x: Proportion | None) -> str:
    if x is None or x.estimate is None or x.low is None or x.high is None:
        return "n/a"
    return (
        f"{x.numerator}/{x.denominator} = {x.estimate:.3f} "
        f"[{x.low:.3f}, {x.high:.3f}] ({x.confidence:.0%} Wilson)"
    )


def _chain(a: Autopsy) -> str:
    links = ", ".join(f"{link.step}:{link.status}" for link in a.links)
    return f"{links}; missing={list(a.missing) or 'none'}; complete={a.complete}"


def build(store: ArtifactStore) -> tuple[Demonstration, AutopsyDemonstration]:
    representation = RepresentationSpec(embedder=HashedNgramEmbedder().spec)
    return demonstration(store, representation), demonstrate(store)


def verify(store: ArtifactStore, digests: tuple[Digest, Digest]) -> tuple[bool, ...]:
    """Store integrity, then both demonstrations rebuilt in an independent store."""
    store.verify()
    with_temp = ArtifactStore(Path(store.root) / "replay")
    again = build(with_temp)
    with_temp.verify()
    return tuple(a.digest == b for a, b in zip(again, digests, strict=True))


def report(mem: Demonstration, ad: AutopsyDemonstration, replayed: tuple[bool, ...]) -> str:
    fm = mem.forgetting_metrics
    out = [
        "# MEMORIA lifecycle demonstration",
        "",
        "> **This is a demonstration, not a benchmark.** It follows one generated world through",
        "> the pipeline, chosen by fixed rules, to make the stored artifacts inspectable. Its",
        "> numbers are single-world illustrations; the measured, multi-seed results are in",
        "> `docs/experiments/`.",
        "",
        "## 1. Experience → memory → retrieval → contradiction",
        f"- world: `{mem.world}`; stages (run digests): "
        + ", ".join(f"{k}=`{v[:19]}`" for k, v in mem.runs),
    ]
    probe, cited, edges, shown, extra = mem.contradiction
    out.append(
        f"- contradiction: probe `{probe}` answered from `{cited[:19]}`; the graph shows "
        f"{edges} contradiction edge(s), retrieval surfaced {shown} counter-memory(ies), and "
        f"{extra} contradicting memory(ies) were visible only in the graph"
        if probe
        else "- contradiction: none found in this world"
    )
    out += [
        "",
        "## 2. Consolidation, forgetting and interference",
        f"- graph before / after targeted forgetting: `{mem.graph_before[:19]}` / "
        f"`{mem.graph_after[:19]}`; diff `{mem.graph_diff[:19]}`",
        f"- forgetting is an availability intervention (nothing is deleted): retention "
        f"{_p(fm.retention)}; precision {_p(fm.precision)}; recall {_p(fm.recall)}",
        f"- probes whose answer changed under forgetting: {len(mem.forgetting_changed)}",
        f"- wrong-memory + contaminated answers before / after interference injection: "
        f"{mem.interference_counts[0]} / {mem.interference_counts[1]}",
        f"- failures introduced / reduced across all stages: {len(mem.introduced_failures)} / "
        f"{len(mem.reduced_failures)}",
        "",
        "## 3. Provenance of an answer given from a derived memory",
        f"- graph `{mem.autopsy.graph[:19]}`; walk from `{mem.autopsy.start[:40]}`: "
        f"{len(mem.autopsy.steps)} steps reaching {len(mem.autopsy.experiences)} experience(s), "
        f"{len(mem.autopsy.sources)} source(s), {len(mem.autopsy.claims)} claim(s), "
        f"{len(mem.autopsy.consolidations)} consolidation(s)",
        "",
        "## 4. Feedback, importance, retention → autopsy",
        f"- adaptive world `{ad.world[:19]}`; memory autopsy of `{ad.memory_autopsy.key}` at day "
        f"{ad.memory_autopsy.at:g}, answer {ad.memory_autopsy.answer!r}:",
        f"  {_chain(ad.memory_autopsy)}",
    ]
    if ad.replay:
        r = ad.replay
        out += [
            "",
            "## 5. Replay around a source correction",
            f"- key `{r.key}`: state of item `{r.item[:24]}` just before / after correction "
            f"`{r.event[:24]}` (recorded at day {r.recorded_at:g}): "
            f"{r.item_before} → {r.item_after}",
            f"- snapshot digests before / after: `{r.before.digest[:19]}` / "
            f"`{r.after.digest[:19]}`",
        ]
    out += ["", "## 6. Counterfactual replay on fixed evidence"]
    for c in ad.counterfactuals:
        out.append(
            f"- {c.label}: {c.changed}/{c.decisions} recorded decisions change "
            f"({_p(c.changed_share)}); gained {c.gained}, lost {c.lost} "
            f"(analysis only, against hidden truth). {c.interpretation}"
        )
    out += [
        "",
        "## 7. Belief revision and calibrated uncertainty → belief autopsy",
        f"- `{ad.belief_autopsy.system}` on `{ad.belief_autopsy.key}` at day "
        f"{ad.belief_autopsy.at:g}, answer {ad.belief_autopsy.answer!r}:",
        f"  {_chain(ad.belief_autopsy)}",
        "",
        "## 8. Verification",
        f"- artifact store re-hashed: ok; memory-lifecycle rebuilt into an independent store: "
        f"{'identical' if replayed[0] else 'DIFFERENT'}; adaptive-lifecycle rebuilt: "
        f"{'identical' if replayed[1] else 'DIFFERENT'}",
        f"- digests: memory lifecycle `{mem.digest}`; adaptive lifecycle `{ad.digest}`",
        "",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0] if args else "var/lifecycle")
    store = ArtifactStore(root / "artifacts")
    mem, ad = build(store)
    replayed = verify(store, (mem.digest, ad.digest))
    text = report(mem, ad, replayed)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text)
    if not all(replayed):
        raise SystemExit("replay produced different digests")


if __name__ == "__main__":
    main()
