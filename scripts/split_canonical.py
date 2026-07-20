"""CLI: assign the train/val/test split over the deduplicated canonical set."""
from __future__ import annotations

import collections

from orthovision.config import resolve_path
from orthovision.labels.schema import read_canonical
from orthovision.split.build import run_split


def main() -> None:
    outputs, summary = run_split()
    print("split counts:", summary["counts"])
    print("assignment sha256:", summary["assignment_sha256"][:16], "...")
    for fold, path in outputs.items():
        pos = collections.Counter()
        recs = read_canonical(path)
        for r in recs:
            for key, on in r.labels.items():
                if on:
                    pos[key] += 1
        print(f"  {fold:<5} n={len(recs):>3}  positives={dict(sorted(pos.items()))}")


if __name__ == "__main__":
    main()
