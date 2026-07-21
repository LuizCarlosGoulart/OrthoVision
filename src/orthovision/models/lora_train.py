"""LoRA fine-tuning of the BiomedCLIP image encoder for multi-label diagnosis.

Freezes the backbone, injects LoRA into the ViT MLP linears, and trains LoRA +
a linear head with BCE. Directly comparable to the F3 linear probe (same head,
frozen vs adapted encoder), isolating the LoRA contribution.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn as nn

from ..labels.schema import PATHOLOGY_KEYS, CanonicalRecord
from .backbone import Backbone
from .lora import DEFAULT_TARGETS, add_lora, lora_parameters


def _iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def train_lora_head(
    backbone: Backbone,
    paths: Sequence[str],
    labels: torch.Tensor,
    *,
    rank: int,
    alpha: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> nn.Linear:
    """Adapt the encoder (LoRA) + fit a head. Returns the trained head; the
    backbone's model is modified in place (LoRA injected, trainable)."""
    torch.manual_seed(seed)
    model = backbone.model
    for p in model.parameters():
        p.requires_grad_(False)
    n_wrapped = add_lora(model.visual, rank, alpha, tuple(targets))
    if n_wrapped == 0:
        raise RuntimeError("no LoRA layers injected; check target names")

    dim = model.encode_image(backbone.image_batch(paths[:1])).shape[1]
    head = nn.Linear(dim, labels.shape[1]).to(backbone.device)

    params = lora_parameters(model.visual) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    head.train()
    order = list(range(len(paths)))
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm = [order[i] for i in torch.randperm(len(order), generator=rng)]
        total = 0.0
        for lo, hi in _iter_batches(len(perm), batch_size):
            idx = perm[lo:hi]
            x = backbone.image_batch([paths[i] for i in idx])
            y = labels[idx].to(backbone.device)
            opt.zero_grad()
            logits = head(model.encode_image(x))
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {epoch + 1}/{epochs}  loss={total / len(perm):.4f}")
    model.eval()
    head.eval()
    return head


@torch.no_grad()
def lora_scores(
    backbone: Backbone,
    head: nn.Linear,
    paths: Sequence[str],
    batch_size: int = 16,
    keys: Sequence[str] = PATHOLOGY_KEYS,
) -> dict[str, list[float]]:
    """Score images with the adapted encoder + head; per-class logits."""
    logits: list[torch.Tensor] = []
    for lo, hi in _iter_batches(len(paths), batch_size):
        x = backbone.image_batch(paths[lo:hi])
        logits.append(head(backbone.model.encode_image(x)))
    all_logits = torch.cat(logits, dim=0)
    return {key: all_logits[:, i].tolist() for i, key in enumerate(keys)}
