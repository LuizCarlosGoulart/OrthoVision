import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from orthovision.ingest.store import ImmutabilityError, ingest_archives


def _png_bytes(size=(800, 400), color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _make_snapshot(tmp_path: Path) -> Path:
    """Build a tiny DENTEX-like snapshot with one zip mirroring the real layout."""
    snap = tmp_path / "snap"
    (snap / "DENTEX").mkdir(parents=True)
    zpath = snap / "DENTEX" / "training_data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("training_data/quadrant-enumeration-disease/xrays/train_0.png", _png_bytes())
        zf.writestr("training_data/quadrant-enumeration-disease/xrays/train_1.png", _png_bytes())
        zf.writestr(
            "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
            b'{"images": []}',
        )
        zf.writestr("training_data/quadrant/xrays/train_0.png", _png_bytes())
        # must be dropped:
        zf.writestr("training_data/unlabelled/xrays/train_0.png", _png_bytes())
        zf.writestr(
            "training_data/quadrant-enumeration-disease/xrays/.ipynb_checkpoints/train_0-checkpoint.png",
            _png_bytes(),
        )
    return snap


ARCHIVES = {
    "training_data.zip": [
        {"include": "training_data/quadrant-enumeration-disease/xrays/", "split": "train", "subset": "diagnosis"},
        {"include": "training_data/quadrant/xrays/", "split": "train", "subset": "quadrant"},
        {
            "include": "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
            "split": "train",
            "subset": "diagnosis",
            "kind": "annotation",
        },
    ]
}
EXCLUDE = [".ipynb_checkpoints/", "unlabelled/"]


def test_extracts_selected_members_with_split_subset(tmp_path):
    snap = _make_snapshot(tmp_path)
    raw = tmp_path / "raw"

    records = ingest_archives(snap, raw, ARCHIVES, EXCLUDE, "dentex", "CC-BY")

    files = {r.file for r in records}
    assert files == {
        "train/diagnosis/train_0.png",
        "train/diagnosis/train_1.png",
        "train/diagnosis/annotations/train_quadrant_enumeration_disease.json",
        "train/quadrant/train_0.png",
    }
    # extracted to disk
    assert (raw / "train/diagnosis/train_0.png").exists()
    assert (raw / "train/diagnosis/annotations/train_quadrant_enumeration_disease.json").exists()


def test_excludes_unlabelled_and_checkpoints(tmp_path):
    snap = _make_snapshot(tmp_path)
    records = ingest_archives(snap, tmp_path / "raw", ARCHIVES, EXCLUDE, "dentex", "CC-BY")
    for r in records:
        assert "unlabelled" not in r.file
        assert ".ipynb_checkpoints" not in r.file


def test_kind_and_metadata_recorded(tmp_path):
    snap = _make_snapshot(tmp_path)
    records = ingest_archives(snap, tmp_path / "raw", ARCHIVES, EXCLUDE, "dentex", "CC-BY")
    by_file = {r.file: r for r in records}
    ann = by_file["train/diagnosis/annotations/train_quadrant_enumeration_disease.json"]
    assert ann.kind == "annotation" and ann.split == "train" and ann.subset == "diagnosis"
    img = by_file["train/quadrant/train_0.png"]
    assert img.kind == "image" and img.subset == "quadrant"


def test_reingest_is_idempotent(tmp_path):
    snap = _make_snapshot(tmp_path)
    raw = tmp_path / "raw"
    r1 = ingest_archives(snap, raw, ARCHIVES, EXCLUDE, "dentex", "CC-BY")
    r2 = ingest_archives(snap, raw, ARCHIVES, EXCLUDE, "dentex", "CC-BY")
    assert [r.file for r in r1] == [r.file for r in r2]


def test_immutability_on_changed_content(tmp_path):
    snap = _make_snapshot(tmp_path)
    raw = tmp_path / "raw"
    ingest_archives(snap, raw, ARCHIVES, EXCLUDE, "dentex", "CC-BY")

    # rebuild the zip with different pixels for the same member name
    zpath = snap / "DENTEX" / "training_data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "training_data/quadrant/xrays/train_0.png", _png_bytes(color=(10, 10, 10))
        )
    with pytest.raises(ImmutabilityError):
        ingest_archives(snap, raw, ARCHIVES, EXCLUDE, "dentex", "CC-BY")
