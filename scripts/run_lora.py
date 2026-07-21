"""CLI: LoRA-adapt the BiomedCLIP image encoder and evaluate on the test fold."""
from __future__ import annotations

import argparse
import time

import torch

from orthovision.config import load_config, resolve_path
from orthovision.datamodule.labels import label_columns, label_matrix
from orthovision.eval.metrics import per_class_metrics
from orthovision.labels.schema import PATHOLOGY_KEYS, read_canonical
from orthovision.models.backbone import load_backbone
from orthovision.models.baselines import _summarize, write_results
from orthovision.models.lora_train import lora_scores, train_lora_head
from orthovision.preprocess.transform import PreprocessConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LoRA adaptation (F4).")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap train images (local CPU).")
    args = ap.parse_args()

    exp = load_config("experiment/lora")["experiment"]
    lc, tc = exp["lora"], exp["train"]
    epochs = args.epochs if args.epochs is not None else tc["epochs"]
    limit = args.limit if args.limit is not None else tc["train_limit"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | epochs: {epochs} | train_limit: {limit}")

    pre = PreprocessConfig.from_config(load_config("preprocess"))
    train = read_canonical(resolve_path("manifests/train.jsonl"))
    test = read_canonical(resolve_path("manifests/test.jsonl"))
    if limit:
        train = train[:limit]

    train_paths = [str(resolve_path(r.image_path)) for r in train]
    test_paths = [str(resolve_path(r.image_path)) for r in test]
    y = torch.tensor(label_matrix(train), dtype=torch.float32)
    test_labels = label_columns(test)

    backbone = load_backbone(load_config("model/biomedclip"), pre, device)
    t0 = time.time()
    head = train_lora_head(
        backbone, train_paths, y,
        rank=lc["rank"], alpha=lc["alpha"], epochs=epochs,
        batch_size=tc["batch_size"], lr=tc["lr"],
        weight_decay=tc["weight_decay"], seed=exp["seed"],
        targets=lc["targets"],
    )
    print(f"  trained in {time.time() - t0:.0f}s")

    scores = lora_scores(backbone, head, test_paths, keys=PATHOLOGY_KEYS)
    summary = _summarize(per_class_metrics(scores, test_labels))
    results = {
        "biomedclip::lora": summary,
        "run_config": {"epochs": epochs, "train_limit": limit, "rank": lc["rank"], "seed": exp["seed"]},
    }
    out = write_results(results)
    print(f"results -> {out}")
    print(f"macroAUC {summary['macro_auc']:.3f} | local {summary['local_macro_auc']:.3f} | global {summary['global_macro_auc']:.3f}")
    print("per-class AUC:", {k: round(v["auc"], 3) for k, v in summary["per_class"].items()})


if __name__ == "__main__":
    main()
