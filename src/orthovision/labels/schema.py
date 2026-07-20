"""Canonical record: one image per record (see docs/dataset-schema.md).

This is the project's contract. Records are produced pre-split in F1; the ``split``
field stays ``None`` until F2 assigns train/val/test on a patient-aware partition.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Canonical multi-label order for the 4 DENTEX pathologies.
PATHOLOGY_KEYS: tuple[str, ...] = (
    "caries",
    "deep_caries",
    "periapical_lesion",
    "impacted_tooth",
)


@dataclass(frozen=True)
class CanonicalRecord:
    record_id: str
    image_path: str                       # repo-relative, openable path
    image_sha256: str
    patient_group_id: str                 # grouping key for the F2 split
    source: str
    modality: str
    specialty: str
    language: str
    labels: dict[str, int]                # multi-hot over PATHOLOGY_KEYS
    label_certainty: dict[str, str]       # per class: certain | uncertain | absent
    pathology_granularity: dict[str, str]  # per class: global | local
    original_taxonomy: dict[str, Any]     # raw source labels (audit)
    reliability: str                      # strong | weak
    text_description: str | None = None
    split: str | None = None              # assigned in F2
    ingest_version: str = "0.0.0"


def write_canonical(records: Iterable[CanonicalRecord], path: str | Path) -> Path:
    """Write canonical records to JSONL, sorted by record_id for determinism."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.record_id)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in ordered:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_canonical(path: str | Path) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CanonicalRecord(**json.loads(line)))
    return records
