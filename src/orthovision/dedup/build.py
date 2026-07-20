"""Run deduplication over the canonical manifest (F1)."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config, resolve_path
from ..labels.build import CANONICAL_MANIFEST
from ..labels.schema import CanonicalRecord, read_canonical, write_canonical
from .dedup import deduplicate
from .phash import dhash

DEDUP_MANIFEST = "manifests/canonical.dentex.dedup.jsonl"
DEDUP_REPORT = "manifests/dedup.report.json"


def run_dedup() -> tuple[Path, dict]:
    cfg = load_config("dedup")["dedup"]
    records = read_canonical(resolve_path(CANONICAL_MANIFEST))

    hash_size = cfg.get("hash_size", 16)
    hashes = {r.record_id: dhash(resolve_path(r.image_path), hash_size) for r in records}
    result = deduplicate(records, hashes, threshold=cfg["hamming_threshold"])

    dedup_path = write_canonical(result.kept, resolve_path(DEDUP_MANIFEST))

    report = {
        "input": len(records),
        "kept": len(result.kept),
        "removed": len(result.removed),
        "groups": result.groups,
        "removed_map": result.removed,
        "hamming_threshold": cfg["hamming_threshold"],
    }
    report_path = resolve_path(DEDUP_REPORT)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return dedup_path, report
