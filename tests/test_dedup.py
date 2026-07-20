import io

from PIL import Image

from orthovision.dedup.dedup import deduplicate
from orthovision.dedup.phash import dhash, hamming
from orthovision.labels.schema import CanonicalRecord


def _rec(rid, reliability="strong"):
    return CanonicalRecord(
        record_id=rid, image_path=f"{rid}.png", image_sha256="x", patient_group_id=rid,
        source="dentex", modality="panoramic_xray", specialty="dentistry", language="en",
        labels={"caries": 0, "deep_caries": 0, "periapical_lesion": 0, "impacted_tooth": 0},
        label_certainty={}, pathology_granularity={}, original_taxonomy={}, reliability=reliability,
    )


def _gradient_png(tmp_path, name, shift=0):
    img = Image.new("L", (64, 64))
    img.putdata([(x + shift) % 256 for _ in range(64) for x in range(64)])
    p = tmp_path / name
    img.save(p)
    return p


def test_hamming_zero_for_identical(tmp_path):
    p = _gradient_png(tmp_path, "a.png")
    assert hamming(dhash(p), dhash(p)) == 0


def test_identical_images_are_deduplicated():
    # same hash for both -> one removed
    recs = [_rec("dentex_train_0"), _rec("dentex_train_1")]
    hashes = {"dentex_train_0": 0xABCD, "dentex_train_1": 0xABCD}
    result = deduplicate(recs, hashes, threshold=5)
    assert len(result.kept) == 1
    assert result.kept[0].record_id == "dentex_train_0"  # lowest id kept
    assert result.removed == {"dentex_train_1": "dentex_train_0"}


def test_distinct_images_are_all_kept():
    recs = [_rec("a"), _rec("b")]
    hashes = {"a": 0x0000, "b": 0xFFFF}  # hamming 16 >> threshold
    result = deduplicate(recs, hashes, threshold=5)
    assert len(result.kept) == 2
    assert result.removed == {}


def test_keeps_stronger_reliability():
    recs = [_rec("weak_one", reliability="weak"), _rec("strong_two", reliability="strong")]
    hashes = {"weak_one": 1, "strong_two": 1}
    result = deduplicate(recs, hashes, threshold=5)
    assert [r.record_id for r in result.kept] == ["strong_two"]
    assert result.removed == {"weak_one": "strong_two"}
