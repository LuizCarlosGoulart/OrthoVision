"""CLI: deduplicate the canonical manifest (perceptual hash)."""
from __future__ import annotations

from orthovision.dedup.build import run_dedup


def main() -> None:
    path, report = run_dedup()
    print(f"deduped manifest -> {path}")
    print(f"input={report['input']} kept={report['kept']} removed={report['removed']}")
    if report["groups"]:
        print("duplicate groups:", report["groups"])


if __name__ == "__main__":
    main()
