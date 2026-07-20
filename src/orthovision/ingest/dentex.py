"""DENTEX acquisition (Hugging Face) and ingestion into the immutable raw store.

The download is a thin, lazily-imported wrapper so the core ingestion logic (and
its tests) never require network access.
"""
from __future__ import annotations

from pathlib import Path

from ..config import load_config, paths, resolve_path
from .manifest import IngestionRecord, write_manifest
from .store import ingest_images


def download_snapshot(repo_id: str, revision: str | None = None) -> Path:
    """Download the DENTEX dataset snapshot; return the local snapshot directory."""
    from huggingface_hub import snapshot_download  # lazy: network dependency

    return Path(
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
    )


def run_ingest(
    *, overwrite: bool = False, src_dir: str | Path | None = None
) -> tuple[Path, list[IngestionRecord]]:
    """Ingest DENTEX into the raw store and write the ingestion manifest."""
    data_cfg = load_config("data/dentex")["dataset"]
    ingest_cfg = load_config("ingest")["ingest"]

    raw_dir = resolve_path(paths()["dentex_raw"])
    if src_dir is None:
        src_dir = download_snapshot(data_cfg["source"]["repo_id"])

    records = ingest_images(
        src_dir,
        raw_dir,
        source=data_cfg["name"],
        license=data_cfg["license"],
        overwrite=overwrite,
    )
    manifest_path = write_manifest(records, resolve_path(ingest_cfg["ingestion_manifest"]))
    return manifest_path, records
