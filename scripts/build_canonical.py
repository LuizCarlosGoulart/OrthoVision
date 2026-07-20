"""CLI: build the canonical manifest for the labeled DENTEX diagnosis set."""
from __future__ import annotations

import collections

from orthovision.labels.build import run_build_canonical


def main() -> None:
    path, records = run_build_canonical()
    positives = collections.Counter()
    for r in records:
        for key, on in r.labels.items():
            if on:
                positives[key] += 1
    print(f"built {len(records)} canonical records -> {path}")
    print("positives per pathology:", dict(sorted(positives.items())))


if __name__ == "__main__":
    main()
