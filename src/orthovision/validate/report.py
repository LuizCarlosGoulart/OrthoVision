"""Validation over the ingested raw store -> conformance report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..config import load_config, paths, resolve_path
from ..ingest.store import iter_images
from .rules import ImageRules, check_image


def validate_dir(src_dir: str | Path, rules: ImageRules) -> tuple[list[Path], dict[str, str]]:
    """Return (passed images, {excluded image path: reason})."""
    passed: list[Path] = []
    excluded: dict[str, str] = {}
    for img in iter_images(src_dir):
        reason = check_image(img, rules)
        if reason is None:
            passed.append(img)
        else:
            excluded[str(img)] = reason
    return passed, excluded


def build_report(passed: list[Path], excluded: dict[str, str]) -> dict:
    reasons = Counter(excluded.values())
    return {
        "total": len(passed) + len(excluded),
        "passed": len(passed),
        "excluded": len(excluded),
        "excluded_by_reason": dict(sorted(reasons.items())),
    }


def run_validate() -> tuple[Path, dict]:
    """Validate the DENTEX raw store and write the conformance report."""
    cfg = load_config("validate")
    rules = ImageRules.from_config(cfg)
    raw_dir = resolve_path(paths()["dentex_raw"])

    passed, excluded = validate_dir(raw_dir, rules)
    report = build_report(passed, excluded)

    report_path = resolve_path(cfg["validate"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path, report
