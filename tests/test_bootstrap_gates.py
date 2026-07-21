import math

from orthovision.eval.bootstrap import bootstrap_auc_ci, paired_delta_ci
from orthovision.eval.gates import check_gates


def test_bootstrap_ci_on_separable_is_high():
    scores = [0.1 * i for i in range(20)]
    labels = [0] * 10 + [1] * 10
    ci = bootstrap_auc_ci(scores, labels, n_boot=200, seed=0)
    assert ci["mean"] > 0.95 and ci["lo"] > 0.8 and ci["lo"] <= ci["hi"] <= 1.0


def test_paired_delta_positive_when_a_better():
    labels = [0] * 10 + [1] * 10
    good = [0.1 * i for i in range(20)]          # perfect ranking
    bad = [0.1 * (20 - i) for i in range(20)]    # inverted
    d = paired_delta_ci(good, bad, labels, n_boot=200, seed=0)
    assert d["lo"] > 0  # a strictly better than b, CI excludes 0


def _summary(aucs: dict[str, float], macro: float) -> dict:
    return {"per_class": {k: {"auc": v} for k, v in aucs.items()}, "macro_auc": macro}


def test_gates_pass_when_lora_beats_baselines():
    keys = {"caries": 0.62, "deep_caries": 0.66, "periapical_lesion": 0.70, "impacted_tooth": 0.72}
    probe = _summary({"caries": 0.55, "deep_caries": 0.63, "periapical_lesion": 0.69, "impacted_tooth": 0.71}, 0.646)
    lora = _summary(keys, 0.675)
    zs_b = _summary({k: 0.5 for k in keys}, 0.514)
    zs_g = _summary({k: 0.5 for k in keys}, 0.551)
    result = check_gates(lora, probe, zs_b, zs_g)
    assert result["pass"] is True


def test_gates_fail_on_caries_regression():
    keys = {"caries": 0.54, "deep_caries": 0.66, "periapical_lesion": 0.70, "impacted_tooth": 0.72}
    probe = _summary({"caries": 0.55, "deep_caries": 0.63, "periapical_lesion": 0.69, "impacted_tooth": 0.71}, 0.646)
    lora = _summary(keys, 0.675)
    zs_b = _summary({k: 0.5 for k in keys}, 0.514)
    zs_g = _summary({k: 0.5 for k in keys}, 0.551)
    result = check_gates(lora, probe, zs_b, zs_g)
    assert result["g3_caries_above_probe"] is False
    assert result["pass"] is False
    assert "caries" in result["g2_regressions"]
