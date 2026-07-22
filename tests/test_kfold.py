from orthovision.labels.schema import CanonicalRecord
from orthovision.split.kfold import kfold_assignment, record_folds


def _rec(rid, pg, caries=0):
    return CanonicalRecord(
        record_id=rid, image_path=f"{rid}.png", image_sha256="x", patient_group_id=pg,
        source="dentex", modality="panoramic_xray", specialty="dentistry", language="en",
        labels={"caries": caries, "deep_caries": 0, "periapical_lesion": 0, "impacted_tooth": 0},
        label_certainty={}, pathology_granularity={}, original_taxonomy={}, reliability="strong",
    )


def test_kfold_covers_all_groups_once():
    recs = [_rec(f"r{i}", f"p{i}", caries=i % 2) for i in range(50)]
    a = kfold_assignment(recs, 5, seed=1)
    assert set(a) == {f"p{i}" for i in range(50)}
    assert set(a.values()) <= set(range(5))


def test_kfold_is_patient_aware():
    recs = []
    for i in range(20):
        recs += [_rec(f"p{i}_a", f"p{i}", i % 2), _rec(f"p{i}_b", f"p{i}", i % 2)]
    folds = record_folds(recs, 5, seed=1)
    fold_of = {r.record_id: folds[j] for j, r in enumerate(recs)}
    for i in range(20):
        assert fold_of[f"p{i}_a"] == fold_of[f"p{i}_b"]


def test_kfold_folds_are_balanced():
    recs = [_rec(f"r{i}", f"p{i}", caries=i % 2) for i in range(100)]
    folds = record_folds(recs, 5, seed=1)
    counts = [folds.count(x) for x in range(5)]
    assert all(15 <= c <= 25 for c in counts)  # ~20 each
