"""CLI: ingest DENTEX into the immutable raw store and write the manifest."""
from __future__ import annotations

import argparse

from orthovision.ingest.dentex import run_ingest


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest DENTEX into the raw store.")
    ap.add_argument(
        "--src-dir",
        default=None,
        help="Local snapshot dir; if omitted, download from Hugging Face.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting raw files with changed content (breaks immutability).",
    )
    args = ap.parse_args()

    path, records = run_ingest(overwrite=args.overwrite, src_dir=args.src_dir)
    print(f"ingested {len(records)} files -> {path}")


if __name__ == "__main__":
    main()
