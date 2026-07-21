"""Evaluation metrics — dependency-free (pure Python).

Multi-label diagnosis is scored per pathology with rank-based metrics that need no
probability calibration: ROC-AUC and Average Precision. Aggregates are macro
(unweighted mean over classes with both positives and negatives present).
"""
from __future__ import annotations

from typing import Sequence

NAN = float("nan")


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, with tie handling.

    Returns NaN if a class has no positives or no negatives.
    """
    n = len(scores)
    n_pos = sum(1 for y in labels if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return NAN

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(ranks[i] for i in range(n) if labels[i])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average Precision (area under precision-recall), interpolation-free.

    Returns NaN if there are no positives.
    """
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return NAN

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            tp += 1
            precision = tp / rank
            ap += precision  # recall increases by 1/n_pos at each positive
    return ap / n_pos


def per_class_metrics(
    scores: dict[str, Sequence[float]], labels: dict[str, Sequence[int]]
) -> dict[str, dict[str, float]]:
    """Return {pathology: {'auc': ..., 'ap': ..., 'n_pos': ...}} for each class."""
    out: dict[str, dict[str, float]] = {}
    for key in scores:
        out[key] = {
            "auc": roc_auc(scores[key], labels[key]),
            "ap": average_precision(scores[key], labels[key]),
            "n_pos": float(sum(1 for y in labels[key] if y)),
        }
    return out


def macro_mean(per_class: dict[str, dict[str, float]], metric: str) -> float:
    """Unweighted mean of a metric over classes where it is defined (non-NaN)."""
    vals = [m[metric] for m in per_class.values() if m[metric] == m[metric]]
    return sum(vals) / len(vals) if vals else NAN
