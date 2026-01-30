# src/ecg_fm/models/ecg_classifiers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

# from ecg_fm.models.registry import register_model


# -----------------------------
# Small reusable components
# -----------------------------


class GlobalPool1D(nn.Module):
    """
    Pool [B, C, T] -> [B, C]
    mode: "mean" | "max" | "meanmax"
    """

    def __init__(self, mode: Literal["mean", "max", "meanmax"] = "mean"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return x.mean(dim=-1)
        if self.mode == "max":
            return x.amax(dim=-1)
        if self.mode == "meanmax":
            return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        raise ValueError(f"Unknown pool mode: {self.mode}")


class ConvBlock1D(nn.Module):
    """Conv1d -> BN -> ReLU (simple and stable)."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int, s: int = 1, p: Optional[int] = None
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def freeze_module(m: nn.Module) -> None:
    """Disable grads for all parameters."""
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module) -> None:
    """Enable grads for all parameters."""
    for p in m.parameters():
        p.requires_grad = True


# -----------------------------
# 1) LSTM + CNN classifier
# -----------------------------


class LSTMCNNClassifier(nn.Module):
    """
    Input:  [B, 1, L]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        stem_ch: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Simple CNN stem: downsample time to reduce LSTM compute
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(stem_ch, stem_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=stem_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        feat_dim = lstm_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        y, _ = self.lstm(x)  # [B, T, H*dir]

        feat = y.mean(dim=1)  # mean pool over time -> [B, H*dir]
        return self.head(feat)


@register_model("lstm_cnn_small")
def build_lstm_cnn_small(num_classes: int, **kwargs) -> nn.Module:
    return LSTMCNNClassifier(num_classes=num_classes)


# -----------------------------
# 2) FM + head (probe / finetune)
# -----------------------------


@dataclass
class FMConfig:
    ckpt_path: str
    feature: Literal["mean", "cls"] = "mean"
    freeze: bool = True
    unfreeze_last_n: int = 0


class FMHeadClassifier(nn.Module):
    """
    Wrap a FM encoder and a classification head.

    Assumption (you can adapt):
      encoder(x) -> features in one of these forms:
        - [B, D]          (global feature)
        - [B, T, D]       (token features)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        feature: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature = feature

        # NOTE: you must set encoder_out_dim correctly when you integrate your FM.
        encoder_out_dim = getattr(encoder, "out_dim", None)
        if encoder_out_dim is None:
            raise ValueError(
                "FM encoder must expose `out_dim` attribute (or adapt this part)."
            )

        self.head = nn.Sequential(
            nn.LayerNorm(encoder_out_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)

        # Handle token features: [B, T, D]
        if feats.dim() == 3:
            if self.feature == "mean":
                feats = feats.mean(dim=1)
            elif self.feature == "cls":
                feats = feats[:, 0]  # assume first token is CLS
            else:
                raise ValueError(f"Unknown FM feature: {self.feature}")

        logits = self.head(feats)
        return logits


def load_fm_encoder(ckpt_path: str):
    """
    Load full ECGMAE_1D model and use its `encode()` as encoder interface.
    """
    import torch

    # from ecg_fm.models.fm.ecg_mae_1d import ECGMAE_1D
    from ecg_fm.models.mini_mae import ECGMAE_1D

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" not in ckpt or "args" not in ckpt:
        raise ValueError("Invalid FM checkpoint format")

    a = ckpt["args"]

    # 1. Rebuild FM model with EXACT same hyperparameters
    fm = ECGMAE_1D(
        n_leads=1,
        patch_size=a["patch_size"],
        d_model=a["d_model"],
        enc_depth=a["depth"],
        dec_depth=2,  # decoder depth does not matter for probe
        n_heads=a["n_heads"],
        dim_ff=a["dim_ff"],
        dropout=a["dropout"],
        mask_ratio=a["mask_ratio"],
        pos_max_len=4096,
    )

    # 2. Load weights
    missing, unexpected = fm.load_state_dict(ckpt["model"], strict=False)
    print(f"[FM load] missing={len(missing)} unexpected={len(unexpected)}")

    # 3. Freeze everything (probe default)
    for p in fm.parameters():
        p.requires_grad = False

    # 4. Wrap encode() as forward()
    class FMEncoderWrapper(torch.nn.Module):
        def __init__(self, fm):
            super().__init__()
            self.fm = fm
            self.out_dim = fm.d_model

        def forward(self, x):
            # x: [B, 1, L]
            return self.fm.encode(x, pool="mean")  # [B, D]

    return FMEncoderWrapper(fm)


def apply_unfreeze_policy(encoder: nn.Module, unfreeze_last_n: int) -> None:
    """
    Optional: unfreeze last N blocks for finetuning.
    This depends on how your encoder is structured (e.g., encoder.blocks list).
    """
    # Default: unfreeze all if unfreeze_last_n <= 0 is not requested explicitly.
    # You should adapt to your FM architecture.
    if unfreeze_last_n <= 0:
        return

    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        # Fallback: unfreeze everything (or raise)
        unfreeze_module(encoder)
        return

    # Freeze all first, then unfreeze last N blocks
    freeze_module(encoder)
    for b in blocks[-unfreeze_last_n:]:
        unfreeze_module(b)


@register_model("fm_probe")
def build_fm_probe(
    *,
    num_classes: int,
    input_len: int,
    fm_ckpt: str,
    fm_feature: str = "mean",
    **kwargs,
) -> nn.Module:
    encoder = load_fm_encoder(fm_ckpt)
    freeze_module(encoder)  # probe: freeze encoder
    return FMHeadClassifier(
        encoder=encoder, num_classes=num_classes, feature=fm_feature
    )


@register_model("fm_finetune")
def build_fm_finetune(
    *,
    num_classes: int,
    input_len: int,
    fm_ckpt: str,
    fm_feature: str = "mean",
    fm_unfreeze_last_n: int = 2,
    **kwargs,
) -> nn.Module:
    encoder = load_fm_encoder(fm_ckpt)

    # finetune policy
    freeze_module(encoder)
    apply_unfreeze_policy(encoder, fm_unfreeze_last_n)

    return FMHeadClassifier(
        encoder=encoder, num_classes=num_classes, feature=fm_feature
    )


# -----------------------------
# 3) 1D-CNN classifier
# -----------------------------


class CNN1D_Small(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)  # [B,128]
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


@register_model("cnn_small")
def build_cnn_small(num_classes: int, **kwargs) -> nn.Module:
    return CNN1D_Small(num_classes=num_classes)
