"""Ingestion manifest: one JSONL record per ingested file.

The manifest is the provenance record for the immutable raw store. It is written
deterministically (records sorted by ``file``) so the same source revision always
yields byte-identical output.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class IngestionRecord:
    file: str      # path relative to the dataset's raw-store root (posix)
    sha256: str
    bytes: int
    source: str
    license: str
    split: str | None = None      # train | test | val
    subset: str | None = None     # diagnosis | quadrant | quadrant_enumeration
    kind: str = "image"           # image | annotation


def write_manifest(records: Iterable[IngestionRecord], path: str | Path) -> Path:
    """Write records to a JSONL manifest, sorted by ``file`` for determinism."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.file)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in ordered:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_manifest(path: str | Path) -> list[IngestionRecord]:
    records: list[IngestionRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(IngestionRecord(**json.loads(line)))
    return records
