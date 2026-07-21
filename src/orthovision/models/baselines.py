"""F3 baseline ladder: generic-CLIP zero-shot, BiomedCLIP zero-shot, BiomedCLIP
linear probe — evaluated on the frozen test fold with per-class AUC/AP."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..config import load_config, resolve_path
from ..datamodule.labels import label_columns, label_matrix
from ..eval.metrics import macro_mean, per_class_metrics
from ..labels.schema import PATHOLOGY_KEYS, read_canonical
from ..preprocess.transform import PreprocessConfig
from .backbone import load_backbone
from .linear_probe import probe_scores, train_linear_probe
from .zero_shot import zero_shot_scores

# Central-hypothesis grouping (matches configs/data/dentex.yaml granularity).
LOCAL_KEYS = ("caries", "deep_caries", "periapical_lesion")
GLOBAL_KEYS = ("impacted_tooth",)


def _summarize(per_class: dict) -> dict:
    return {
        "per_class": per_class,
        "macro_auc": macro_mean(per_class, "auc"),
        "macro_ap": macro_mean(per_class, "ap"),
        "local_macro_auc": macro_mean({k: per_class[k] for k in LOCAL_KEYS}, "auc"),
        "global_macro_auc": macro_mean({k: per_class[k] for k in GLOBAL_KEYS}, "auc"),
    }


def run_baselines(device: str = "cpu", limit: int | None = None) -> dict:
    exp = load_config("experiment/base")["experiment"]
    pre = PreprocessConfig.from_config(load_config("preprocess"))
    class_prompts = exp["class_prompts"]

    train = read_canonical(resolve_path("manifests/train.jsonl"))
    test = read_canonical(resolve_path("manifests/test.jsonl"))
    if limit:
        train, test = train[:limit], test[:limit]

    train_paths = [str(resolve_path(r.image_path)) for r in train]
    test_paths = [str(resolve_path(r.image_path)) for r in test]
    test_labels = label_columns(test)

    results: dict[str, dict] = {}
    for cfg_name in ("clip_vitb16", "biomedclip"):
        t0 = time.time()
        backbone = load_backbone(load_config(f"model/{cfg_name}"), pre, device)
        test_feats = backbone.encode_images(test_paths)

        zs = zero_shot_scores(backbone, class_prompts, test_feats)
        results[f"{cfg_name}::zero_shot"] = _summarize(per_class_metrics(zs, test_labels))

        if cfg_name == "biomedclip":
            train_feats = backbone.encode_images(train_paths)
            y = torch.tensor(label_matrix(train), dtype=torch.float32)
            probe = train_linear_probe(train_feats, y, seed=exp["seed"])
            ps = probe_scores(probe, test_feats, PATHOLOGY_KEYS)
            results[f"{cfg_name}::linear_probe"] = _summarize(per_class_metrics(ps, test_labels))

        print(f"  [{cfg_name}] done in {time.time() - t0:.0f}s")
        del backbone

    return results


def write_results(results: dict) -> Path:
    run_dir = resolve_path("experiments") / f"baselines_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "baselines.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, indent=2)
    return out
