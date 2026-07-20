from orthovision.ingest.manifest import IngestionRecord
from orthovision.labels.dentex_coco import build_diagnosis_records, parse_coco_diagnosis
from orthovision.labels.schema import PATHOLOGY_KEYS

LABEL_MAP = {
    "Impacted": "impacted_tooth",
    "Caries": "caries",
    "Periapical Lesion": "periapical_lesion",
    "Deep Caries": "deep_caries",
}
GRANULARITY = {
    "caries": "local",
    "deep_caries": "local",
    "periapical_lesion": "local",
    "impacted_tooth": "global",
}

# Minimal COCO: 2 images. img1 has Caries + Impacted (two teeth); img2 has nothing.
COCO = {
    "images": [
        {"id": 1, "file_name": "train_0.png", "height": 1316, "width": 2744},
        {"id": 2, "file_name": "train_1.png", "height": 1316, "width": 2744},
    ],
    "annotations": [
        {"image_id": 1, "category_id_3": 1},  # Caries
        {"image_id": 1, "category_id_3": 0},  # Impacted
    ],
    "categories_3": [
        {"id": 0, "name": "Impacted"},
        {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Periapical Lesion"},
        {"id": 3, "name": "Deep Caries"},
    ],
}

INGESTION = [
    IngestionRecord("train_0.png", "hash0", 10, "dentex", "CC-BY", "train", "diagnosis", "image"),
    IngestionRecord("train_1.png", "hash1", 11, "dentex", "CC-BY", "train", "diagnosis", "image"),
    # noise that must be ignored:
    IngestionRecord("train_0.png", "annh", 5, "dentex", "CC-BY", "train", "diagnosis", "annotation"),
    IngestionRecord("train_0.png", "qhash", 9, "dentex", "CC-BY", "train", "quadrant", "image"),
]


def test_parse_unions_diseases_per_image():
    per_image = parse_coco_diagnosis(COCO, LABEL_MAP)
    assert per_image["train_0.png"] == {"caries", "impacted_tooth"}
    assert per_image["train_1.png"] == set()


def test_build_multihot_and_certainty():
    recs = {r.record_id: r for r in build_diagnosis_records(
        COCO, INGESTION, LABEL_MAP, GRANULARITY,
        dentex_raw="data/raw/dentex", source="dentex", modality="panoramic_xray",
        specialty="dentistry", language="en",
    )}
    r0 = recs["dentex_train_train_0"]
    assert r0.labels == {"caries": 1, "deep_caries": 0, "periapical_lesion": 0, "impacted_tooth": 1}
    assert r0.label_certainty["caries"] == "certain"
    assert r0.label_certainty["deep_caries"] == "absent"
    assert r0.pathology_granularity == GRANULARITY
    # references the raw image (not the annotation, not the quadrant subset)
    assert r0.image_path == "data/raw/dentex/train_0.png"
    assert r0.image_sha256 == "hash0"
    assert r0.split is None and r0.reliability == "strong"
    assert set(r0.labels) == set(PATHOLOGY_KEYS)


def test_negative_image_is_all_absent():
    recs = {r.record_id: r for r in build_diagnosis_records(
        COCO, INGESTION, LABEL_MAP, GRANULARITY,
        dentex_raw="data/raw/dentex", source="dentex", modality="panoramic_xray",
        specialty="dentistry", language="en",
    )}
    r1 = recs["dentex_train_train_1"]
    assert sum(r1.labels.values()) == 0
    assert all(v == "absent" for v in r1.label_certainty.values())
