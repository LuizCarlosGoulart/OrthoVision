"""Iterative multi-label stratification (Sechidis et al., 2011).

Standard approach for splitting multi-label data so every fold preserves each
label's proportion — essential here because periapical_lesion is rare (116/705)
and naive splitting can leave a fold with none. Dependency-free.

Items with no positive label are stratified via a virtual NONE label so the
all-negative images also spread proportionally.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping

NONE_LABEL = "__none__"


def _choose_fold(
    folds: list[str],
    desired_label: dict[str, float],
    desired_fold: dict[str, float],
    rng: random.Random,
) -> str:
    """Fold that most needs this label; tie-break by overall need, then random."""
    best = max(desired_label[f] for f in folds)
    cand = [f for f in folds if desired_label[f] == best]
    if len(cand) > 1:
        top = max(desired_fold[f] for f in cand)
        cand = [f for f in cand if desired_fold[f] == top]
    if len(cand) > 1:
        cand = [cand[rng.randrange(len(cand))]]
    return cand[0]


def iterative_stratify(
    items: list[str],
    label_sets: Mapping[str, set[str]],
    ratios: Mapping[str, float],
    seed: int = 1337,
) -> dict[str, str]:
    """Assign each item to a fold. Returns ``{item: fold_name}``, deterministic."""
    rng = random.Random(seed)
    folds = list(ratios.keys())
    total = sum(ratios.values())
    frac = {f: ratios[f] / total for f in folds}

    labels_of = {
        it: (set(label_sets.get(it) or ()) or {NONE_LABEL}) for it in items
    }

    label_count: dict[str, int] = defaultdict(int)
    for it in items:
        for lab in labels_of[it]:
            label_count[lab] += 1

    desired_fold = {f: frac[f] * len(items) for f in folds}
    desired = {
        lab: {f: frac[f] * label_count[lab] for f in folds} for lab in label_count
    }

    assignment: dict[str, str] = {}
    remaining = set(items)
    rem_label = dict(label_count)

    while remaining:
        active = [
            lab
            for lab, n in rem_label.items()
            if n > 0 and any(lab in labels_of[it] for it in remaining)
        ]
        if not active:
            break
        # rarest remaining label first; deterministic tie-break by name
        target = min(active, key=lambda l: (rem_label[l], l))

        for it in sorted(it for it in remaining if target in labels_of[it]):
            fold = _choose_fold(folds, desired[target], desired_fold, rng)
            assignment[it] = fold
            remaining.discard(it)
            desired_fold[fold] -= 1
            for lab in labels_of[it]:
                desired[lab][fold] -= 1
                rem_label[lab] -= 1

    return assignment
