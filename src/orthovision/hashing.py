"""File hashing utilities (content integrity + provenance)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 of an in-memory buffer (e.g. a zip member)."""
    return hashlib.sha256(data).hexdigest()
