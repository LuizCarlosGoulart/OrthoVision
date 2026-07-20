from orthovision.split.stratify import iterative_stratify

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _synthetic(n_common=200, n_rare=15, n_none=20):
    labels = {}
    for i in range(n_common):
        labels[f"c{i}"] = {"caries"}
    for i in range(n_rare):
        labels[f"r{i}"] = {"periapical_lesion"}
    for i in range(n_none):
        labels[f"n{i}"] = set()
    return labels


def test_assigns_every_item_once():
    labels = _synthetic()
    a = iterative_stratify(list(labels), labels, RATIOS, seed=1337)
    assert set(a) == set(labels)
    assert set(a.values()) <= set(RATIOS)


def test_deterministic_with_seed():
    labels = _synthetic()
    a = iterative_stratify(list(labels), labels, RATIOS, seed=1337)
    b = iterative_stratify(list(labels), labels, RATIOS, seed=1337)
    assert a == b


def test_rare_label_present_in_every_fold():
    labels = _synthetic(n_rare=15)  # 15 >= 3 folds, so each fold should get some
    a = iterative_stratify(list(labels), labels, RATIOS, seed=1337)
    per_fold = {f: 0 for f in RATIOS}
    for item, fold in a.items():
        if labels[item] == {"periapical_lesion"}:
            per_fold[fold] += 1
    assert all(count > 0 for count in per_fold.values()), per_fold


def test_proportions_are_approximately_respected():
    labels = _synthetic(n_common=700, n_rare=0, n_none=0)
    a = iterative_stratify(list(labels), labels, RATIOS, seed=1337)
    counts = {f: sum(1 for v in a.values() if v == f) for f in RATIOS}
    assert abs(counts["train"] - 490) <= 10
    assert abs(counts["val"] - 105) <= 10
    assert abs(counts["test"] - 105) <= 10
