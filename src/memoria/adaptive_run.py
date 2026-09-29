"""Runs the Super-Phase 5 study: ``python -m memoria.adaptive_run var/superphase5 [--quick]``.

Stores every run, the study and its performance record; re-simulates the runs of the first and
last replicate into an independent store and compares digests; prints the report.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from memoria.adaptive_report import report
from memoria.adaptive_study import reproduce, run_study, study_spec
from memoria.artifacts import ArtifactStore


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    quick = "--quick" in args
    spec = study_spec(store, 3, 2, ("changing", "adversarial")) if quick else study_spec(store)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    study, perf = run_study(spec, store, say)
    fresh = ArtifactStore(str(store.root) + "-fresh")
    rep = reproduce(study, spec, fresh, sorted({0, spec.replicates - 1}))
    say(f"reproduction {rep.identical}/{rep.runs_checked}")
    print(f"spec          {study.spec}")
    print(f"study         {study.digest}")
    print(f"performance   {perf.digest}")
    print(f"reproduction  {rep.model_dump_json()}")
    print()
    print(report(study))


if __name__ == "__main__":
    main()
