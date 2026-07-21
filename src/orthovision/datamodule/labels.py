"""Torch-free helpers to turn canonical records into label matrices."""
from __future__ import annotations

from typing import Sequence

from ..labels.schema import PATHOLOGY_KEYS, CanonicalRecord


def label_columns(
    records: Sequence[CanonicalRecord], keys: Sequence[str] = PATHOLOGY_KEYS
) -> dict[str, list[int]]:
    """Return {pathology: [0/1 per record]} aligned to ``records`` order."""
    return {k: [int(r.labels[k]) for r in records] for k in keys}


def label_matrix(
    records: Sequence[CanonicalRecord], keys: Sequence[str] = PATHOLOGY_KEYS
) -> list[list[int]]:
    """Return a row-per-record multi-hot matrix in ``keys`` order."""
    return [[int(r.labels[k]) for k in keys] for r in records]
