"""Map the DENTEX train COCO annotation to canonical records.

The DENTEX diagnosis annotation is per-tooth detection with three hierarchical
category axes; ``category_id_3`` is the disease. The image-level multi-label is
the union of disease classes over that image's tooth annotations.

Only the train COCO is mapped here: it is the clean, expert-annotated 4-class
source. The packaged test/val per-image annotations use a divergent clinical
vocabulary that does not correspond to the 4 classes and are intentionally not
mapped (see the F1 notes / pending ADR).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..ingest.manifest import IngestionRecord
from .schema import PATHOLOGY_KEYS, CanonicalRecord


def parse_coco_diagnosis(
    coco: Mapping[str, Any], label_map: Mapping[str, str]
) -> dict[str, set[str]]:
    """Return ``{file_name: {canonical disease keys present}}`` for every image."""
    id_to_name = {c["id"]: c["name"] for c in coco["categories_3"]}
    img_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}

    per_image: dict[str, set[str]] = {im["file_name"]: set() for im in coco["images"]}
    for ann in coco["annotations"]:
        disease_name = id_to_name[ann["category_id_3"]]
        per_image[img_id_to_file[ann["image_id"]]].add(label_map[disease_name])
    return per_image


def _index_images(
    ingestion: list[IngestionRecord], split: str, subset: str
) -> dict[str, IngestionRecord]:
    """Index ingested images by basename within a (split, subset)."""
    return {
        Path(r.file).name: r
        for r in ingestion
        if r.kind == "image" and r.split == split and r.subset == subset
    }


def build_diagnosis_records(
    coco: Mapping[str, Any],
    ingestion: list[IngestionRecord],
    label_map: Mapping[str, str],
    granularity: Mapping[str, str],
    *,
    dentex_raw: str,
    source: str,
    modality: str,
    specialty: str,
    language: str,
    reliability: str = "strong",
    ingest_version: str = "0.0.0",
    dentex_split: str = "train",
    subset: str = "diagnosis",
) -> list[CanonicalRecord]:
    """Build canonical records for the labeled DENTEX diagnosis images."""
    per_image = parse_coco_diagnosis(coco, label_map)
    index = _index_images(ingestion, dentex_split, subset)

    records: list[CanonicalRecord] = []
    for file_name, present in sorted(per_image.items()):
        ing = index[file_name]  # KeyError => ingestion/annotation mismatch (a real bug)
        labels = {k: (1 if k in present else 0) for k in PATHOLOGY_KEYS}
        certainty = {k: ("certain" if labels[k] else "absent") for k in PATHOLOGY_KEYS}
        granularity_map = {k: granularity[k] for k in PATHOLOGY_KEYS}
        record_id = f"{source}_{dentex_split}_{Path(file_name).stem}"

        records.append(
            CanonicalRecord(
                record_id=record_id,
                image_path=f"{dentex_raw}/{ing.file}",
                image_sha256=ing.sha256,
                patient_group_id=record_id,  # 1 panoramic = 1 patient (assumption)
                source=source,
                modality=modality,
                specialty=specialty,
                language=language,
                labels=labels,
                label_certainty=certainty,
                pathology_granularity=granularity_map,
                original_taxonomy={
                    "dentex_split": dentex_split,
                    "disease_classes": sorted(present),
                },
                reliability=reliability,
                split=None,  # assigned in F2
                ingest_version=ingest_version,
            )
        )
    return records
