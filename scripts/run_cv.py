"""CLI: k-fold cross-validation with pooled out-of-fold evaluation."""
from __future__ import annotations

import argparse
import json
import time

import torch

from orthovision.config import resolve_path
from orthovision.labels.schema import PATHOLOGY_KEYS
from orthovision.models.cv import run_cv


def main() -> None:
    ap = argparse.ArgumentParser(description="K-fold cross-validation (LoRA vs baselines).")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap images (smoke test).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    res = run_cv(k=args.k, device=device, epochs=args.epochs, limit=args.limit)

    out = resolve_path("experiments") / f"cv_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out / 'cv.json'}")

    print("\nmacro AUC (pooled out-of-fold, all", res["n"], "images):")
    for m in ("zs_generic", "zs_biomed", "probe", "lora"):
        s = res["summaries"][m]
        print(f"  {m:<11} macro {s['macro_auc']:.3f}  local {s['local_macro_auc']:.3f}  global {s['global_macro_auc']:.3f}")
    print("\nper-class LoRA AUC with 95% CI:")
    for key in PATHOLOGY_KEYS:
        ci = res["lora_per_class_ci"][key]
        auc = res["summaries"]["lora"]["per_class"][key]["auc"]
        print(f"  {key:<18} {auc:.3f}  CI[{ci['lo']:.3f}, {ci['hi']:.3f}]")
    d = res["caries_lora_minus_probe_ci"]
    print(f"\ncaries LoRA-probe delta {d['mean']:+.3f}  CI[{d['lo']:+.3f}, {d['hi']:+.3f}]"
          + ("  (significant)" if d["lo"] > 0 else "  (includes 0)"))
    print("GATES:", res["gates"])


if __name__ == "__main__":
    main()
