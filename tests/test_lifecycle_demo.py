"""The lifecycle demonstration reproduces itself and links every answer to its evidence."""

from pathlib import Path

from memoria.artifacts import ArtifactStore
from memoria.lifecycle_demo import build, report, verify


def test_lifecycle_demo_reproduces_and_every_autopsy_is_complete(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    mem, ad = build(store)
    replayed = verify(store, (mem.digest, ad.digest))
    assert replayed == (True, True)  # rebuilt in an independent store, same digests
    for autopsy in (ad.memory_autopsy, ad.belief_autopsy):
        assert autopsy.complete
        assert not autopsy.missing
    text = report(mem, ad, replayed)
    assert "demonstration, not a benchmark" in text
    assert "DIFFERENT" not in text
