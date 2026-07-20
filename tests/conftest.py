"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image(tmp_path: Path):
    """Factory that writes a synthetic image under tmp_path and returns its path."""

    def _make(
        name: str,
        size: tuple[int, int] = (800, 400),
        color: tuple[int, int, int] = (120, 120, 120),
        fmt: str = "PNG",
    ) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(p, format=fmt)
        return p

    return _make
