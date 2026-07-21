"""Light tests for the model logic (torch only; no open_clip / no downloads)."""
from __future__ import annotations

import torch

from orthovision.eval.metrics import roc_auc
from orthovision.models.linear_probe import probe_scores, train_linear_probe
from orthovision.models.zero_shot import zero_shot_scores


def test_linear_probe_fits_separable_features():
    torch.manual_seed(0)
    feats = torch.randn(64, 8)
    y = torch.zeros(64, 2)
    y[:, 0] = (feats[:, 0] > 0).float()
    y[:, 1] = (feats[:, 1] > 0).float()

    probe = train_linear_probe(feats, y, seed=0)
    scores = probe_scores(probe, feats, ["a", "b"])
    auc_a = roc_auc(scores["a"], [int(v) for v in y[:, 0].tolist()])
    assert auc_a > 0.9


class _StubBackbone:
    def encode_texts(self, texts):
        # one orthonormal text vector per class, in prompt order
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def test_zero_shot_scores_rank_by_similarity():
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # img0 ~ class a, img1 ~ class b
    scores = zero_shot_scores(_StubBackbone(), {"a": "pa", "b": "pb"}, feats)
    assert scores["a"][0] > scores["a"][1]
    assert scores["b"][1] > scores["b"][0]
