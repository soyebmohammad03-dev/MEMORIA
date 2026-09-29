"""Runs the Super-Phase 6 benchmark: ``python -m memoria.benchmark_run var/bench [--quick]``.

Stores the manifest, every run and the result; re-runs a representative set of replicates into an
independent store and compares run digests; runs the autopsy demonstration; prints the report.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from memoria.artifacts import ArtifactStore
from memoria.autopsy_demo import demonstrate
from memoria.benchmark import (
    BenchmarkManifest,
    BenchResult,
    make_manifest,
    reproduce,
    run_benchmark,
)
from memoria.benchmark_report import report


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    quick = "--quick" in args
    manifest = make_manifest(store, 2, 1, 2, 1) if quick else make_manifest(store)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    result, perf = run_benchmark(manifest, store, say)
    n, nb = manifest.memory_seeds[2], manifest.belief_seeds[2]
    mem_reps = sorted({0, n // 2, n - 1})
    bel_reps = sorted({0, nb - 1})
    say(f"reproducing memory replicates {mem_reps} and belief replicates {bel_reps}")
    fresh = ArtifactStore(str(store.root) + "-fresh")
    rep = reproduce(result, manifest, store, fresh, mem_reps, bel_reps)
    say(f"reproduction {rep.identical}/{rep.runs_checked}")
    demo = demonstrate(store)
    fresh_demo = demonstrate(fresh)
    print(f"manifest      {result.manifest}")
    print(f"result        {result.digest}")
    print(f"performance   {perf.digest}")
    print(f"reproduction  {rep.model_dump_json()}")
    print(f"autopsy demo  {demo.digest} (independent store: {fresh_demo.digest})")
    print()
    m = store.get_record(BenchmarkManifest, result.manifest)
    print(report(m, store.get_record(BenchResult, result.digest), demo))


if __name__ == "__main__":
    main()
