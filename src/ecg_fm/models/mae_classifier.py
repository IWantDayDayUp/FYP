# src/models/mae_classifier.py
from __future__ import annotations
import torch
import torch.nn as nn

from .mae_ecg import ECGMAE_1D


class ECGEncoderForClassification(nn.Module):
    """
    Use MAE encoder as a backbone for classification (no masking, no decoder).
    """

    def __init__(self, mae: ECGMAE_1D, n_classes: int, pool: str = "mean"):
        super().__init__()
        self.mae = mae
        self.pool = pool
        self.classifier = nn.Linear(mae.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        returns logits: [B, n_classes]
        """
        # 1) patch embedding
        tokens = self.mae.patch_embed(x)  # [B, T, D]
        tokens = self.mae.pos_embed_enc(tokens)  # [B, T, D]

        # 2) encoder
        h = tokens
        for blk in self.mae.encoder:
            h = blk(h)  # [B, T, D]

        # 3) pooling
        if self.pool == "mean":
            feat = h.mean(dim=1)  # [B, D]
        elif self.pool == "max":
            feat = h.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # 4) classifier head
        logits = self.classifier(feat)
        return logits


def freeze_encoder(model: ECGEncoderForClassification):
    for p in model.mae.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: ECGEncoderForClassification):
    for p in model.mae.parameters():
        p.requires_grad = True
