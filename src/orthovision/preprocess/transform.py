"""Deterministic image normalization (PIL-only).

Steps, per configs/preprocess.yaml:
  1. intensity windowing to [p_low, p_high] percentiles, rescaled to full range;
  2. grayscale -> RGB (3 channels);
  3. resize to a square target, preserving aspect ratio with padding.

Kept dependency-free (no numpy) so F1 has no heavy runtime deps. Augmentation and
tensor conversion belong to the datamodule, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    p_low: float
    p_high: float
    target: int
    keep_aspect_ratio: bool
    pad: bool

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PreprocessConfig":
        pp = cfg["preprocess"]
        return cls(
            p_low=pp["intensity"]["p_low"],
            p_high=pp["intensity"]["p_high"],
            target=pp["resize"]["target"],
            keep_aspect_ratio=pp["resize"]["keep_aspect_ratio"],
            pad=pp["resize"]["pad"],
        )


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def window_intensity(gray: Image.Image, p_low: float, p_high: float) -> Image.Image:
    """Clip to [p_low, p_high] percentiles and rescale to 0..255."""
    values = sorted(gray.getdata())
    lo = _percentile(values, p_low)
    hi = _percentile(values, p_high)
    if hi <= lo:
        return gray
    scale = 255.0 / (hi - lo)
    lut = [0 if v <= lo else 255 if v >= hi else round((v - lo) * scale) for v in range(256)]
    return gray.point(lut)


def resize_pad(img: Image.Image, target: int, keep_aspect: bool, pad: bool) -> Image.Image:
    """Resize to target x target; preserve aspect ratio with padding if requested."""
    if not keep_aspect:
        return img.resize((target, target), Image.LANCZOS)
    w, h = img.size
    scale = target / max(w, h)
    new = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    if not pad:
        return new
    canvas = Image.new(img.mode, (target, target), 0)
    canvas.paste(new, ((target - new.size[0]) // 2, (target - new.size[1]) // 2))
    return canvas


def normalize_image(path: str | Path, cfg: PreprocessConfig) -> Image.Image:
    """Apply the full normalization pipeline; return an RGB target x target image."""
    with Image.open(path) as im:
        gray = im.convert("L")
    windowed = window_intensity(gray, cfg.p_low, cfg.p_high)
    rgb = windowed.convert("RGB")
    return resize_pad(rgb, cfg.target, cfg.keep_aspect_ratio, cfg.pad)
