"""Zero-shot multi-label scoring: cosine similarity of image to per-class prompt."""
from __future__ import annotations

from typing import Sequence

import torch

from .backbone import Backbone


def zero_shot_scores(
    backbone: Backbone,
    class_prompts: dict[str, str],
    image_features: torch.Tensor,
) -> dict[str, list[float]]:
    """Return {pathology: [score per image]} from precomputed image features."""
    keys = list(class_prompts)
    text_feats = backbone.encode_texts([class_prompts[k] for k in keys])  # [C, D]
    sims = image_features @ text_feats.t()  # [N, C]
    return {key: sims[:, i].tolist() for i, key in enumerate(keys)}
