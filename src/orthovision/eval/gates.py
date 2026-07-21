"""ADD §7 approval gates for a candidate adapted model vs the F3 baselines.

Consumes the ``_summarize`` dicts (per_class + macro fields) produced by
models.baselines. Gate 4 (ECE not worse than zero-shot) is not evaluated here —
ECE/calibration is not yet implemented (tracked as a carried gap).
"""
from __future__ import annotations

from typing import Any


def check_gates(
    lora: dict[str, Any],
    probe: dict[str, Any],
    zs_biomed: dict[str, Any],
    zs_generic: dict[str, Any],
) -> dict[str, Any]:
    """Return per-gate booleans + an overall pass (gates 1-3; gate 4 pending)."""
    g1 = lora["macro_auc"] > zs_biomed["macro_auc"] and lora["macro_auc"] > zs_generic["macro_auc"]

    regressions = [
        k
        for k in lora["per_class"]
        if lora["per_class"][k]["auc"] < probe["per_class"][k]["auc"]
    ]
    g2 = len(regressions) == 0

    g3 = lora["per_class"]["caries"]["auc"] > probe["per_class"]["caries"]["auc"]

    return {
        "g1_macro_above_zeroshot": g1,
        "g2_no_regression_vs_probe": g2,
        "g2_regressions": regressions,
        "g3_caries_above_probe": g3,
        "g4_ece_not_worse": "not_evaluated",
        "pass": bool(g1 and g2 and g3),
    }
