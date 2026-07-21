"""Open-CLIP backbone wrapper with domain-aware image preprocessing.

The default CLIP preprocess uses Resize + CenterCrop, which crops the sides of a
(wide) panoramic radiograph — exactly where molars and impactions sit. We instead
apply our aspect-preserving pad (preprocess.transform.resize_pad) after domain
intensity windowing, then normalize with the model's own mean/std. Encoding is
streamed in batches and returns L2-normalized embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from ..preprocess.transform import PreprocessConfig, resize_pad, window_intensity

_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def _extract_norm(preprocess) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    mean, std, size = _OPENAI_MEAN, _OPENAI_STD, 224
    for t in getattr(preprocess, "transforms", []):
        name = type(t).__name__
        if name == "Normalize":
            mean = tuple(float(x) for x in t.mean)
            std = tuple(float(x) for x in t.std)
        elif name in ("Resize", "CenterCrop"):
            s = getattr(t, "size", None)
            if isinstance(s, int):
                size = s
            elif isinstance(s, (tuple, list)):
                size = int(s[0])
    return mean, std, size


@dataclass
class Backbone:
    name: str
    model: object
    tokenizer: object
    mean: tuple[float, ...]
    std: tuple[float, ...]
    size: int
    pre: PreprocessConfig
    device: str = "cpu"

    def _to_tensor(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as im:
            gray = im.convert("L")
        windowed = window_intensity(gray, self.pre.p_low, self.pre.p_high)
        rgb = resize_pad(windowed.convert("RGB"), self.size, keep_aspect=True, pad=True)
        data = list(rgb.getdata())  # row-major list of (r, g, b)
        t = torch.tensor(data, dtype=torch.float32).reshape(self.size, self.size, 3)
        t = t.permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return (t - mean) / std

    def image_batch(self, paths: Sequence[str | Path]) -> torch.Tensor:
        """Stack preprocessed image tensors for ``paths`` (grad-enabled path)."""
        return torch.stack([self._to_tensor(p) for p in paths]).to(self.device)

    @torch.no_grad()
    def encode_images(self, paths: Sequence[str | Path], batch_size: int = 16) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([self._to_tensor(p) for p in paths[start : start + batch_size]])
            emb = self.model.encode_image(batch.to(self.device))
            feats.append(emb / emb.norm(dim=-1, keepdim=True))
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        emb = self.model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)


def load_backbone(model_cfg: dict, pre: PreprocessConfig, device: str = "cpu") -> Backbone:
    """Load an open_clip backbone from a configs/model/*.yaml ``model`` dict."""
    import open_clip

    m = model_cfg["model"]
    if "hf_hub" in m:
        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{m['hf_hub']}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{m['hf_hub']}")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            m["arch"], pretrained=m["pretrained"]
        )
        tokenizer = open_clip.get_tokenizer(m["arch"])

    model.eval().to(device)
    mean, std, size = _extract_norm(preprocess)
    return Backbone(m["name"], model, tokenizer, mean, std, size, pre, device)
