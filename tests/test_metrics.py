import math

from orthovision.eval.metrics import (
    average_precision,
    macro_mean,
    per_class_metrics,
    roc_auc,
)


def test_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == 1.0


def test_auc_inverted_is_zero():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == 0.0


def test_auc_random_half():
    # positives at the extremes, negatives in the middle -> half the pairs correct
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [1, 0, 0, 1]
    assert roc_auc(scores, labels) == 0.5


def test_auc_handles_ties():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert roc_auc(scores, labels) == 0.5


def test_auc_undefined_without_both_classes():
    assert math.isnan(roc_auc([0.1, 0.2], [0, 0]))
    assert math.isnan(roc_auc([0.1, 0.2], [1, 1]))


def test_average_precision_perfect():
    assert average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0


def test_average_precision_undefined_without_positives():
    assert math.isnan(average_precision([0.1, 0.2], [0, 0]))


def test_macro_mean_skips_nan():
    per_class = {
        "a": {"auc": 1.0}, "b": {"auc": 0.0}, "c": {"auc": float("nan")},
    }
    assert macro_mean(per_class, "auc") == 0.5


def test_per_class_shape():
    scores = {"caries": [0.9, 0.1], "impacted_tooth": [0.2, 0.8]}
    labels = {"caries": [1, 0], "impacted_tooth": [0, 1]}
    m = per_class_metrics(scores, labels)
    assert m["caries"]["auc"] == 1.0 and m["caries"]["n_pos"] == 1.0
