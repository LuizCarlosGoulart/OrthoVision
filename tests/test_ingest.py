import pytest

from orthovision.ingest.manifest import read_manifest, write_manifest
from orthovision.ingest.store import ImmutabilityError, ingest_images


def test_records_all_images_with_hash(tmp_path, make_image):
    src, raw = tmp_path / "src", tmp_path / "raw"
    make_image("src/one.png")
    make_image("src/sub/two.png")

    records = ingest_images(src, raw, source="dentex", license="CC-BY")

    assert len(records) == 2
    assert {r.file for r in records} == {"one.png", "sub/two.png"}
    assert all(r.sha256 and r.bytes > 0 for r in records)
    assert (raw / "one.png").exists() and (raw / "sub" / "two.png").exists()


def test_manifest_is_deterministic_and_sorted(tmp_path, make_image):
    src, raw = tmp_path / "src", tmp_path / "raw"
    make_image("src/b.png")
    make_image("src/a.png")

    m1 = write_manifest(ingest_images(src, raw, "dentex", "CC-BY"), tmp_path / "m1.jsonl")
    # re-ingesting an unchanged source is idempotent -> byte-identical manifest
    m2 = write_manifest(ingest_images(src, raw, "dentex", "CC-BY"), tmp_path / "m2.jsonl")
    assert m1.read_text(encoding="utf-8") == m2.read_text(encoding="utf-8")

    files = [r.file for r in read_manifest(m1)]
    assert files == sorted(files)


def test_immutability_blocks_changed_content(tmp_path, make_image):
    src, raw = tmp_path / "src", tmp_path / "raw"
    make_image("src/one.png", color=(120, 120, 120))
    ingest_images(src, raw, "dentex", "CC-BY")

    # same path, different pixels -> different hash than the stored raw file
    make_image("src/one.png", color=(10, 10, 10))
    with pytest.raises(ImmutabilityError):
        ingest_images(src, raw, "dentex", "CC-BY")
