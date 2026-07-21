"""Multi-label linear probe over frozen CLIP image features.

A single linear layer trained with BCE — the standard strong baseline above
zero-shot. The backbone stays frozen; only the probe is fit, so image features
can be cached (they are here, computed once per fold).
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..labels.schema import PATHOLOGY_KEYS


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 1337,
) -> torch.nn.Linear:
    """Fit a Linear(D, n_classes) with BCEWithLogitsLoss on frozen features."""
    torch.manual_seed(seed)
    n_classes = labels.shape[1]
    probe = torch.nn.Linear(features.shape[1], n_classes).to(features.device)
    labels = labels.to(features.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(features), labels)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def probe_scores(
    probe: torch.nn.Linear,
    features: torch.Tensor,
    keys: Sequence[str] = PATHOLOGY_KEYS,
) -> dict[str, list[float]]:
    """Return {pathology: [logit per image]} (rank-equivalent to probability)."""
    logits = probe(features)
    return {key: logits[:, i].tolist() for i, key in enumerate(keys)}
