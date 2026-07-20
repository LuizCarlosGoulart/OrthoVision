"""Perceptual hashing (difference hash) — dependency-free (PIL only).

dHash resizes to (size+1) x size grayscale and encodes, per row, whether each
pixel is brighter than its right neighbor, yielding a size*size-bit integer.
Near-duplicate images have a small Hamming distance between their hashes.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def dhash(path: str | Path, size: int = 16) -> int:
    with Image.open(path) as im:
        small = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(small.getdata())
    width = size + 1
    value = 0
    for row in range(size):
        for col in range(size):
            left = px[row * width + col]
            right = px[row * width + col + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
