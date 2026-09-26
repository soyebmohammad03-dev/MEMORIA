import os
from pathlib import Path

import pytest

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import Experience, content_hash
from memoria.scenarios import day

EXP = Experience(source="s", content="hello", occurred_at=day(0))


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "var")


def _file(store: ArtifactStore, digest: str) -> Path:
    h = digest.removeprefix("sha256:")
    return store.root / "objects" / h[:2] / h[2:]


def test_put_get_roundtrip_is_content_addressed(store: ArtifactStore) -> None:
    digest = store.put(b"abc")
    assert digest == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert store.get(digest) == b"abc"
    assert store.has(digest)


def test_record_artifact_digest_is_record_digest(store: ArtifactStore) -> None:
    digest = store.put_record(EXP)
    assert digest == EXP.digest == content_hash(EXP.model_dump(mode="json"))
    assert store.get_record(Experience, digest) == EXP


def test_write_once_and_idempotent(store: ArtifactStore) -> None:
    first = store.put(b"x", kind="k")
    assert store.put(b"x", kind="k") == first
    assert store.digests("k") == [first]  # indexed once
    assert not os.access(_file(store, first), os.W_OK)  # read-only on disk


def test_empty_artifact(store: ArtifactStore) -> None:
    assert store.get(store.put(b"")) == b""


def test_index_lists_by_kind_in_write_order(store: ArtifactStore) -> None:
    a, b = store.put(b"a", kind="k"), store.put(b"b", kind="other")
    c = store.put(b"c", kind="k")
    assert store.digests("k") == [a, c]
    assert store.digests("other") == [b]
    assert store.digests("none") == []
    assert ArtifactStore(store.root / "fresh").digests("k") == []


def test_missing_and_malformed_digests(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactIntegrityError, match="not found"):
        store.get("sha256:" + "0" * 64)
    for bad in ("sha256:xyz", "md5:" + "0" * 64, "sha256:" + "A" * 64, "../../etc/passwd"):
        with pytest.raises(ValueError, match="not a sha256"):
            store.get(bad)


def test_corruption_is_detected_on_read_and_write(store: ArtifactStore) -> None:
    digest = store.put(b"original")
    path = _file(store, digest)
    path.chmod(0o644)
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        store.get(digest)
    with pytest.raises(ArtifactIntegrityError):  # never silently "repaired" by a re-put
        store.put(b"original")
    with pytest.raises(ArtifactIntegrityError):
        store.verify()


def test_record_of_wrong_type_or_non_canonical_bytes(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactIntegrityError, match="not a Experience"):
        store.get_record(Experience, store.put(b'{"x": 1}'))
    spaced = EXP.canonical().replace(",", ", ").encode()
    with pytest.raises(ArtifactIntegrityError, match="not canonical"):
        store.get_record(Experience, store.put(spaced))


def test_verify_detects_missing_indexed_artifact_and_partial_writes(
    store: ArtifactStore,
) -> None:
    digest = store.put(b"gone")
    store.verify()
    path = _file(store, digest)
    path.chmod(0o644)
    path.unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        store.verify()
    store.put(b"gone")  # restoring identical content is legitimate
    store.verify()
    (path.parent / f".{path.name}.1.tmp").write_bytes(b"half")
    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        store.verify()


def test_reopen_sees_existing_artifacts(store: ArtifactStore) -> None:
    digest = store.put_record(EXP)
    again = ArtifactStore(store.root)
    assert again.get_record(Experience, digest) == EXP
    assert again.digests("Experience") == [digest]
