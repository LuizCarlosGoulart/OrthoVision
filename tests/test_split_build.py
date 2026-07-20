"""Leakage + integrity tests for the split builder, on a synthetic canonical set."""
from __future__ import annotations

import dataclasses

from orthovision.labels.schema import CanonicalRecord
from orthovision.split.build import _group_labels
from orthovision.split.stratify import iterative_stratify


def _rec(rid, pgid, caries=0):
    return CanonicalRecord(
        record_id=rid, image_path=f"{rid}.png", image_sha256="x", patient_group_id=pgid,
        source="dentex", modality="panoramic_xray", specialty="dentistry", language="en",
        labels={"caries": caries, "deep_caries": 0, "periapical_lesion": 0, "impacted_tooth": 0},
        label_certainty={}, pathology_granularity={}, original_taxonomy={}, reliability="strong",
    )


def test_group_labels_unions_by_patient():
    # two images of the same patient; one has caries -> group is positive
    recs = [_rec("a1", "p1", caries=1), _rec("a2", "p1", caries=0), _rec("b1", "p2", caries=0)]
    groups, labels = _group_labels(recs)
    assert set(groups) == {"p1", "p2"}
    assert labels["p1"] == {"caries"}
    assert labels["p2"] == set()


def test_no_patient_group_spans_two_folds():
    # 30 patients, 2 images each; the split is at group granularity
    recs = []
    for i in range(30):
        pg = f"p{i}"
        recs += [_rec(f"{pg}_a", pg, caries=i % 2), _rec(f"{pg}_b", pg, caries=i % 2)]
    groups, labels = _group_labels(recs)
    assignment = iterative_stratify(sorted(groups), labels, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1337)

    fold_of_record = {}
    for r in recs:
        fold_of_record[r.record_id] = assignment[r.patient_group_id]
    # both images of each patient must land in the same fold
    for i in range(30):
        assert fold_of_record[f"p{i}_a"] == fold_of_record[f"p{i}_b"]
