"""Immutable raw-store ingestion.

Copies source images into the raw store once and records a checksum per file.
The store is immutable: re-ingesting an unchanged source is idempotent, but a
source whose content differs from an already-stored file is a hard error unless
``overwrite`` is explicitly requested (which the F0 policy sets to False).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..hashing import sha256_bytes, sha256_file
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


def _write_immutable(dst: Path, data: bytes, digest: str, overwrite: bool) -> None:
    """Write bytes to the raw store, enforcing immutability of existing files."""
    if dst.exists():
        if sha256_file(dst) != digest:
            if not overwrite:
                raise ImmutabilityError(
                    f"raw file already exists with different content: {dst}"
                )
            dst.write_bytes(data)
        # identical content -> idempotent
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)


def _match_rule(member: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first rule whose ``include`` prefix matches the zip member."""
    for rule in rules:
        if member.startswith(rule["include"]):
            return rule
    return None


def _find_archive(snapshot_dir: str | Path, name: str) -> Path:
    for p in Path(snapshot_dir).rglob(name):
        if p.is_file():
            return p
    raise FileNotFoundError(f"archive not found in snapshot: {name}")


def ingest_archives(
    snapshot_dir: str | Path,
    raw_dir: str | Path,
    archives: dict[str, list[dict[str, Any]]],
    exclude: Iterable[str],
    source: str,
    license: str,
    *,
    overwrite: bool = False,
) -> list[IngestionRecord]:
    """Extract selected members from the dataset zips into the immutable raw store.

    ``archives`` maps a zip basename to a list of rules
    ``{include: <path-prefix>, split, subset, kind?}``. A member is ingested when
    it matches an include prefix, is not excluded, and matches the rule ``kind``
    (image extensions for ``image``; ``.json`` for ``annotation``). Files land at
    ``<split>/<subset>/[annotations/]<basename>`` under ``raw_dir``.
    """
    raw_dir = Path(raw_dir)
    exclude = list(exclude or [])
    records: list[IngestionRecord] = []

    for zip_name, rules in archives.items():
        archive = _find_archive(snapshot_dir, zip_name)
        with zipfile.ZipFile(archive) as zf:
            for member in sorted(zf.namelist()):
                if member.endswith("/"):
                    continue
                if any(token in member for token in exclude):
                    continue
                rule = _match_rule(member, rules)
                if rule is None:
                    continue

                kind = rule.get("kind", "image")
                ext = Path(member).suffix.lower()
                if kind == "image" and ext not in IMAGE_EXTS:
                    continue
                if kind == "annotation" and ext != ".json":
                    continue

                data = zf.read(member)
                digest = sha256_bytes(data)
                rel = Path(rule["split"]) / rule["subset"]
                if kind == "annotation":
                    rel = rel / "annotations"
                rel = rel / Path(member).name
                _write_immutable(raw_dir / rel, data, digest, overwrite)

                records.append(
                    IngestionRecord(
                        file=rel.as_posix(),
                        sha256=digest,
                        bytes=len(data),
                        source=source,
                        license=license,
                        split=rule["split"],
                        subset=rule["subset"],
                        kind=kind,
                    )
                )

    return records
