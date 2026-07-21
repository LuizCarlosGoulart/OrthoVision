import collections

import torch
import torch.nn as nn

from orthovision.models.lora import LoRALinear, add_lora, lora_parameters


def test_lora_starts_as_identity():
    base = nn.Linear(6, 4)
    lora = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 6)
    assert torch.allclose(lora(x), base(x))  # B is zero-initialized


def test_only_ab_are_trainable():
    lora = LoRALinear(nn.Linear(6, 4), rank=2, alpha=4)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}


def test_add_lora_wraps_named_targets():
    mlp = nn.Sequential(
        collections.OrderedDict(
            [("fc1", nn.Linear(4, 8)), ("gelu", nn.GELU()), ("fc2", nn.Linear(8, 4))]
        )
    )
    n = add_lora(mlp, rank=2, alpha=4, targets=("fc1", "fc2"))
    assert n == 2
    assert len(lora_parameters(mlp)) == 4  # A,B per wrapped layer
    # forward still works and starts as identity
    x = torch.randn(2, 4)
    assert mlp(x).shape == (2, 4)
