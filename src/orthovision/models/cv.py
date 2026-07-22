"""K-fold cross-validation over all 705 canonical images (out-of-fold evaluation).

For each fold as the held-out test set, frozen-feature methods (zero-shot, linear
probe) and LoRA are trained on the remaining folds and predict the held-out fold.
Pooling the out-of-fold predictions gives per-class AUC + bootstrap CIs on the full
705 — far more statistical power than a single 103-image test fold.

Compute note: LoRA is retrained per fold (encoder adapts), so k backbone trainings.
Zero-shot / probe reuse features encoded once per backbone. For LoRA early stopping,
a small validation slice is carved from the training folds (not the held-out test
fold), so the test fold stays clean and the scheme works for any k.
"""
from __future__ import annotations

import random

import torch

from ..config import load_config, paths as cfg_paths, resolve_path
from ..datamodule.labels import label_columns, label_matrix
from ..dedup.build import DEDUP_MANIFEST
from ..eval.bootstrap import bootstrap_auc_ci, paired_delta_ci
from ..eval.gates import check_gates
from ..eval.metrics import per_class_metrics
from ..labels.schema import PATHOLOGY_KEYS, read_canonical
from ..models.backbone import load_backbone
from ..models.baselines import _summarize
from ..models.linear_probe import probe_scores, train_linear_probe
from ..models.lora_train import lora_scores, train_lora_head
from ..models.zero_shot import zero_shot_scores
from ..preprocess.transform import PreprocessConfig
from ..split.kfold import record_folds


def _fill(oof: dict, keys, per_class_scores: dict, positions: list[int]) -> None:
    for key in keys:
        for j, pos in enumerate(positions):
            oof[key][pos] = per_class_scores[key][j]


def run_cv(*, k: int = 5, device: str = "cpu", epochs: int | None = None, limit: int | None = None) -> dict:
    pre = PreprocessConfig.from_config(load_config("preprocess"))
    exp = load_config("experiment/lora")["experiment"]
    lc, tc = exp["lora"], exp["train"]
    epochs = epochs if epochs is not None else tc["epochs"]
    seed = exp["seed"]
    class_prompts = load_config("experiment/base")["experiment"]["class_prompts"]

    records = read_canonical(resolve_path(DEDUP_MANIFEST))
    if limit:
        records = records[:limit]
    n = len(records)
    paths = [str(resolve_path(r.image_path)) for r in records]
    labels_cols = label_columns(records)
    folds = record_folds(records, k, seed)
    print(f"CV: {n} images, k={k}, fold sizes={[folds.count(f) for f in range(k)]}")

    methods = ("zs_generic", "zs_biomed", "probe", "lora")
    oof = {m: {key: [None] * n for key in PATHOLOGY_KEYS} for m in methods}

    # ---- generic CLIP zero-shot (fold-independent) ----
    bb = load_backbone(load_config("model/clip_vitb16"), pre, device)
    feats = bb.encode_images(paths)
    _fill(oof["zs_generic"], PATHOLOGY_KEYS, zero_shot_scores(bb, class_prompts, feats), list(range(n)))
    del bb, feats

    # ---- BiomedCLIP zero-shot + linear probe (encode once, probe per fold) ----
    bb = load_backbone(load_config("model/biomedclip"), pre, device)
    feats = bb.encode_images(paths)
    _fill(oof["zs_biomed"], PATHOLOGY_KEYS, zero_shot_scores(bb, class_prompts, feats), list(range(n)))
    y_all = torch.tensor(label_matrix(records), dtype=torch.float32).to(feats.device)
    for f in range(k):
        tr = [i for i in range(n) if folds[i] != f]
        te = [i for i in range(n) if folds[i] == f]
        probe = train_linear_probe(feats[tr], y_all[tr], seed=seed)
        _fill(oof["probe"], PATHOLOGY_KEYS, probe_scores(probe, feats[te], PATHOLOGY_KEYS), te)
    del bb, feats

    # ---- LoRA per fold (fresh backbone; carve a val slice from train for
    #      early stopping — works for any k and keeps the test fold untouched) ----
    val_frac = 0.15
    for f in range(k):
        te = [i for i in range(n) if folds[i] == f]
        non_test = [i for i in range(n) if folds[i] != f]
        shuffled = non_test[:]
        random.Random(seed + f).shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        va = sorted(shuffled[:n_val])
        tr = sorted(shuffled[n_val:])
        print(f"  [fold {f}] train={len(tr)} val={len(va)} test={len(te)}")
        bb = load_backbone(load_config("model/biomedclip"), pre, device)
        y_tr = torch.tensor([[records[i].labels[key] for key in PATHOLOGY_KEYS] for i in tr], dtype=torch.float32).to(device)
        val_labels = {key: [labels_cols[key][i] for i in va] for key in PATHOLOGY_KEYS}
        head = train_lora_head(
            bb, [paths[i] for i in tr], y_tr,
            rank=lc["rank"], alpha=lc["alpha"], epochs=epochs,
            batch_size=tc["batch_size"], lr=tc["lr"], weight_decay=tc["weight_decay"],
            seed=seed, targets=lc["targets"], augment=True,
            val_paths=[paths[i] for i in va], val_labels=val_labels,
        )
        _fill(oof["lora"], PATHOLOGY_KEYS, lora_scores(bb, head, [paths[i] for i in te], keys=PATHOLOGY_KEYS), te)
        del bb, head

    # ---- pooled out-of-fold metrics + CIs + gates on all n images ----
    summaries = {m: _summarize(per_class_metrics(oof[m], labels_cols)) for m in methods}
    caries_delta = paired_delta_ci(oof["lora"]["caries"], oof["probe"]["caries"], labels_cols["caries"], seed=seed)
    lora_ci = {key: bootstrap_auc_ci(oof["lora"][key], labels_cols[key], seed=seed) for key in PATHOLOGY_KEYS}
    gates = check_gates(summaries["lora"], summaries["probe"], summaries["zs_biomed"], summaries["zs_generic"])

    return {
        "k": k, "n": n, "epochs": epochs,
        "summaries": summaries,
        "lora_per_class_ci": lora_ci,
        "caries_lora_minus_probe_ci": caries_delta,
        "gates": gates,
        "config": {"rank": lc["rank"], "alpha": lc["alpha"], "batch_size": tc["batch_size"], "lr": tc["lr"], "seed": seed},
    }
