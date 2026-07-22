"""K-fold patient-aware, multi-label-stratified cross-validation folds.

Reuses the F2 iterative stratification with k equal partitions, grouping by
``patient_group_id`` so no patient spans two folds. Deterministic given the seed.
"""
from __future__ import annotations

from typing import Sequence

from ..labels.schema import CanonicalRecord
from .build import _group_labels
from .stratify import iterative_stratify


def kfold_assignment(records: Sequence[CanonicalRecord], k: int, seed: int = 1337) -> dict[str, int]:
    """Return ``{patient_group_id: fold_index}`` for k folds."""
    groups, labels = _group_labels(list(records))
    ratios = {i: 1.0 / k for i in range(k)}
    return iterative_stratify(sorted(groups), labels, ratios, seed)


def record_folds(records: Sequence[CanonicalRecord], k: int, seed: int = 1337) -> list[int]:
    """Return the fold index per record, aligned to ``records`` order."""
    assignment = kfold_assignment(records, k, seed)
    return [assignment[r.patient_group_id] for r in records]
