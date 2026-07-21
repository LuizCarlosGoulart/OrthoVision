"""Bootstrap confidence intervals for AUC (percentile method, dependency-free)."""
from __future__ import annotations

import random
from typing import Sequence

from .metrics import roc_auc


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def bootstrap_auc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict[str, float]:
    """Percentile bootstrap CI for a single class's ROC-AUC."""
    rng = random.Random(seed)
    n = len(scores)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = roc_auc([scores[i] for i in idx], [labels[i] for i in idx])
        if a == a:  # skip resamples with a single class
            stats.append(a)
    stats.sort()
    mean = sum(stats) / len(stats) if stats else float("nan")
    return {"mean": mean, "lo": _percentile(stats, alpha / 2), "hi": _percentile(stats, 1 - alpha / 2)}


def paired_delta_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict[str, float]:
    """Paired bootstrap CI for the AUC difference (a - b) on the same labels.

    If the CI excludes 0, the difference is significant at the given alpha.
    """
    rng = random.Random(seed)
    n = len(labels)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        lab = [labels[i] for i in idx]
        aa = roc_auc([scores_a[i] for i in idx], lab)
        ab = roc_auc([scores_b[i] for i in idx], lab)
        if aa == aa and ab == ab:
            stats.append(aa - ab)
    stats.sort()
    mean = sum(stats) / len(stats) if stats else float("nan")
    return {"mean": mean, "lo": _percentile(stats, alpha / 2), "hi": _percentile(stats, 1 - alpha / 2)}
