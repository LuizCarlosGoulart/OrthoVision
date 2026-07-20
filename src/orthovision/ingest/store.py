"""Immutable raw-store ingestion.

Copies source images into the raw store once and records a checksum per file.
The store is immutable: re-ingesting an unchanged source is idempotent, but a
source whose content differs from an already-stored file is a hard error unless
``overwrite`` is explicitly requested (which the F0 policy sets to False).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Iterator

from ..hashing import sha256_file
from .manifest import IngestionRecord

IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


class ImmutabilityError(RuntimeError):
    """Raised when an ingest would overwrite an existing raw file with new content."""


def iter_images(src_dir: str | Path, exts: Iterable[str] = IMAGE_EXTS) -> Iterator[Path]:
    """Yield image files under ``src_dir`` in a stable, sorted order."""
    exts = {e.lower() for e in exts}
    for p in sorted(Path(src_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def ingest_images(
    src_dir: str | Path,
    raw_dir: str | Path,
    source: str,
    license: str,
    *,
    overwrite: bool = False,
    exts: Iterable[str] = IMAGE_EXTS,
) -> list[IngestionRecord]:
    """Ingest images from ``src_dir`` into ``raw_dir``; return manifest records."""
    src_dir = Path(src_dir)
    raw_dir = Path(raw_dir)
    records: list[IngestionRecord] = []

    for src in iter_images(src_dir, exts):
        rel = src.relative_to(src_dir)
        dst = raw_dir / rel
        digest = sha256_file(src)

        if dst.exists():
            existing = sha256_file(dst)
            if existing != digest:
                if not overwrite:
                    raise ImmutabilityError(
                        f"raw file already exists with different content: {dst}"
                    )
                dst.write_bytes(src.read_bytes())
            # identical content -> idempotent, nothing to write
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        records.append(
            IngestionRecord(
                file=rel.as_posix(),
                sha256=digest,
                bytes=src.stat().st_size,
                source=source,
                license=license,
            )
        )

    return records
