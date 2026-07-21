"""LoRA fine-tuning of the BiomedCLIP image encoder for multi-label diagnosis.

Freezes the backbone, injects LoRA into the ViT MLP linears, and trains LoRA +
a linear head with BCE. Directly comparable to the F3 linear probe (same head,
frozen vs adapted encoder), isolating the LoRA contribution.
"""
from __future__ import annotations

import copy
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ..eval.metrics import macro_mean, per_class_metrics
from ..labels.schema import PATHOLOGY_KEYS
from .backbone import Backbone
from .lora import DEFAULT_TARGETS, add_lora, lora_parameters


def _iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def _maybe_flip(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Randomly horizontal-flip each image in the batch (presence is flip-invariant)."""
    mask = torch.rand(x.shape[0], generator=rng) < 0.5
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    if idx:
        x[idx] = torch.flip(x[idx], dims=[-1])
    return x


def _snapshot(visual: nn.Module, head: nn.Linear):
    return ([p.detach().clone() for p in lora_parameters(visual)], copy.deepcopy(head.state_dict()))


def _restore(visual: nn.Module, head: nn.Linear, snap) -> None:
    for p, saved in zip(lora_parameters(visual), snap[0]):
        p.data.copy_(saved)
    head.load_state_dict(snap[1])


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
    augment: bool = True,
    val_paths: Sequence[str] | None = None,
    val_labels: Mapping[str, Sequence[int]] | None = None,
) -> nn.Linear:
    """Adapt the encoder (LoRA) + fit a head. Returns the trained head; the
    backbone's model is modified in place (LoRA injected, trainable).

    With ``augment`` each training batch is randomly horizontal-flipped. If
    ``val_paths``/``val_labels`` are given, the epoch with the best validation
    macro-AUC is selected (early stopping), which curbs overfitting on the small
    training set.
    """
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
    rng = torch.Generator().manual_seed(seed)

    best_val = -1.0
    best_snap = None
    for epoch in range(epochs):
        model.train()
        head.train()
        perm = torch.randperm(len(paths), generator=rng).tolist()
        total = 0.0
        for lo, hi in _iter_batches(len(perm), batch_size):
            idx = perm[lo:hi]
            x = backbone.image_batch([paths[i] for i in idx])
            if augment:
                x = _maybe_flip(x, rng)
            y = labels[idx].to(backbone.device)
            opt.zero_grad()
            loss = loss_fn(head(model.encode_image(x)), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        msg = f"    epoch {epoch + 1}/{epochs}  loss={total / len(perm):.4f}"
        if val_paths is not None and val_labels is not None:
            val_auc = macro_mean(
                per_class_metrics(lora_scores(backbone, head, val_paths), val_labels), "auc"
            )
            msg += f"  val_macroAUC={val_auc:.4f}"
            if val_auc > best_val:
                best_val = val_auc
                best_snap = _snapshot(model.visual, head)
                msg += " *"
        print(msg)

    if best_snap is not None:
        _restore(model.visual, head, best_snap)
        print(f"  restored best epoch (val_macroAUC={best_val:.4f})")

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
