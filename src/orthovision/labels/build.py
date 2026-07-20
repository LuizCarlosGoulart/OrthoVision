"""Build the canonical manifest for the labeled DENTEX diagnosis set (F1)."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config, paths, resolve_path
from ..ingest.manifest import read_manifest
from .dentex_coco import build_diagnosis_records
from .schema import CanonicalRecord, write_canonical

CANONICAL_MANIFEST = "manifests/canonical.dentex.jsonl"
TRAIN_COCO = "train/diagnosis/annotations/train_quadrant_enumeration_disease.json"


def run_build_canonical() -> tuple[Path, list[CanonicalRecord]]:
    data_cfg = load_config("data/dentex")["dataset"]
    ingest_cfg = load_config("ingest")["ingest"]

    raw_dir = resolve_path(paths()["dentex_raw"])
    ingestion = read_manifest(resolve_path(ingest_cfg["ingestion_manifest"]))

    with open(raw_dir / TRAIN_COCO, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    granularity = {p["key"]: p["granularity"] for p in data_cfg["pathologies"]}
    records = build_diagnosis_records(
        coco,
        ingestion,
        label_map=data_cfg["diagnosis_label_map"],
        granularity=granularity,
        dentex_raw=paths()["dentex_raw"],
        source=data_cfg["name"],
        modality=data_cfg["modality"],
        specialty=data_cfg["specialty"],
        language=data_cfg["language"],
        reliability=data_cfg["reliability"],
    )
    out = write_canonical(records, resolve_path(CANONICAL_MANIFEST))
    return out, records
