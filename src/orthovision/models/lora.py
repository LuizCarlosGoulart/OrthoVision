"""Minimal LoRA for the CLIP image encoder (dependency-free beyond torch).

Wraps selected ``nn.Linear`` layers with a frozen base + a low-rank update
``B @ A`` (B zero-initialized, so training starts as the identity). Targets the
MLP linears (``c_fc``, ``c_proj``) of the ViT residual blocks: they are called as
plain modules, unlike the packed ``nn.MultiheadAttention`` in/out projections
(which are accessed by ``.weight`` and would break if wrapped).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# timm ViT (BiomedCLIP visual): attention qkv/proj + MLP fc1/fc2 are plain
# nn.Linear called as modules, so all are safe LoRA targets.
DEFAULT_TARGETS = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # Create the low-rank factors on the base layer's device/dtype so LoRA
        # works even when injected after the model is moved to GPU.
        device, dtype = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(
    module: nn.Module, rank: int, alpha: float, targets=DEFAULT_TARGETS
) -> int:
    """Recursively wrap target nn.Linear layers with LoRA. Returns count wrapped."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += add_lora(child, rank, alpha, targets)
    return count


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The trainable LoRA parameters (A/B) under a module."""
    params: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params += [m.A, m.B]
    return params
