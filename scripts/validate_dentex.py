"""CLI: validate the ingested DENTEX raw store and write the conformance report."""
from __future__ import annotations

import json

from orthovision.validate.report import run_validate


def main() -> None:
    path, report = run_validate()
    print(f"validation report -> {path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
