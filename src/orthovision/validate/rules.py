"""Image conformance rules (fail-early gate after ingestion)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ImageRules:
    min_width: int
    min_height: int
    allowed_formats: frozenset[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ImageRules":
        img = cfg["validate"]["image"]
        width, height = img["min_resolution"]
        formats = frozenset(fmt.lower() for fmt in img["allowed_formats"])
        return cls(min_width=width, min_height=height, allowed_formats=formats)


def check_image(path: str | Path, rules: ImageRules) -> str | None:
    """Return ``None`` if the image passes, else a short failure reason.

    Reasons: ``"format"``, ``"unreadable"``, ``"resolution"``.
    """
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in rules.allowed_formats:
        return "format"
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncation/corruption
        with Image.open(path) as im:  # fresh handle: verify() consumes the file
            width, height = im.size
    except Exception:
        return "unreadable"
    if width < rules.min_width or height < rules.min_height:
        return "resolution"
    return None
