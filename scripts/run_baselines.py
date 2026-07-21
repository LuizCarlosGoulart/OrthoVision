"""CLI: run the F3 baseline ladder on the frozen test fold."""
from __future__ import annotations

import argparse

import torch

from orthovision.models.baselines import run_baselines, write_results


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run zero-shot + linear-probe baselines.")
    ap.add_argument("--limit", type=int, default=None, help="Cap images per fold (smoke test).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    results = run_baselines(device=device, limit=args.limit)
    out = write_results(results)

    print(f"\nresults -> {out}\n")
    header = f"{'model::strategy':<28} {'macroAUC':>9} {'local':>7} {'global':>7}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<28} {_fmt(r['macro_auc']):>9} {_fmt(r['local_macro_auc']):>7} {_fmt(r['global_macro_auc']):>7}")
    print("\nper-class AUC:")
    for name, r in results.items():
        aucs = {k: round(v["auc"], 3) for k, v in r["per_class"].items()}
        print(f"  {name:<28} {aucs}")


if __name__ == "__main__":
    main()
