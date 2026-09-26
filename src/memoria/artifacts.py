"""Content-addressed, write-once local artifact store (constitution I10).

Layout under ``root`` (by default a git-ignored ``var/`` directory):

- ``objects/<hh>/<rest-of-sha256>``: artifact bytes, named by their SHA-256, read-only.
- ``index.jsonl``: one canonical line ``{"digest", "kind"}`` per artifact, in first-write
  order. The index only aids discovery; objects are authoritative.

A record artifact's bytes are its canonical JSON, so its artifact digest equals its
record digest. Nothing is ever overwritten: writing existing content is a no-op after
verification, and every read re-hashes the bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError

from memoria.core import Record, canonical_json


class ArtifactIntegrityError(RuntimeError):
    """An artifact is missing, corrupted, or not what it claims to be."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.jsonl"

    def _path(self, digest: str) -> Path:
        hexdigest = digest.removeprefix("sha256:")
        if len(hexdigest) != 64 or not all(c in "0123456789abcdef" for c in hexdigest):
            raise ValueError(f"not a sha256 digest: {digest!r}")
        return self.root / "objects" / hexdigest[:2] / hexdigest[2:]

    def put(self, data: bytes, kind: str = "blob") -> str:
        """Store ``data`` once; return its digest. Idempotent for identical content."""
        digest = _sha256(data)
        path = self._path(digest)
        if path.exists():
            self.get(digest)  # verifies the existing copy; never overwritten
            return digest
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.chmod(0o444)
        os.replace(tmp, path)  # atomic: readers see all of the artifact or none of it
        with self._index.open("a", encoding="utf-8") as f:
            f.write(canonical_json({"digest": digest, "kind": kind}) + "\n")
        return digest

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as e:
            raise ArtifactIntegrityError(f"artifact {digest} not found") from e
        if _sha256(data) != digest:
            raise ArtifactIntegrityError(f"artifact {digest}: content does not match its digest")
        return data

    def has(self, digest: str) -> bool:
        return self._path(digest).exists()

    def put_record(self, record: Record) -> str:
        return self.put(record.canonical().encode("utf-8"), kind=type(record).__name__)

    def get_record[R: Record](self, model: type[R], digest: str) -> R:
        try:
            record = model.model_validate_json(self.get(digest))
        except ValidationError as e:
            raise ArtifactIntegrityError(f"artifact {digest} is not a {model.__name__}") from e
        if record.digest != digest:  # e.g. non-canonical bytes that parse to other content
            raise ArtifactIntegrityError(f"artifact {digest} is not canonical {model.__name__}")
        return record

    def digests(self, kind: str) -> list[str]:
        """Digests of artifacts of ``kind``, in first-write order."""
        if not self._index.exists():
            return []
        entries = (json.loads(line) for line in self._index.read_text("utf-8").splitlines())
        return [e["digest"] for e in entries if e["kind"] == kind]

    def verify(self) -> None:
        """Re-hash every object and check that every indexed artifact exists."""
        for path in (self.root / "objects").glob("*/*"):
            if path.name.startswith("."):
                raise ArtifactIntegrityError(f"incomplete write left behind: {path}")
            self.get("sha256:" + path.parent.name + path.name)
        if self._index.exists():
            for line in self._index.read_text("utf-8").splitlines():
                digest = json.loads(line)["digest"]
                if not self.has(digest):
                    raise ArtifactIntegrityError(f"indexed artifact {digest} is missing")
