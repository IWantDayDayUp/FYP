import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    """
    1D patch embedding using Conv1d.
    Input:  x [B, N, L]
    Output: tokens [B, T, D], where T = L // P
    """

    def __init__(self, n_leads: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            n_leads, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.proj(x)  # [B, D, T]
        return t.transpose(1, 2)  # [B, T, D]


class MLP(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block (no causal mask).
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)

        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.drop2(y)
        return x


class SinCosPosEmbed(nn.Module):
    """
    Fixed sinusoidal positional embedding.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


def random_masking_with_ids(x: torch.Tensor, mask_ratio: float):
    """
    Standard MAE-style random masking.

    Args:
        x: [B, T, D]
    Returns:
        x_keep: [B, T_keep, D]
        mask: [B, T] 0=keep, 1=mask
        ids_restore: [B, T]
        ids_keep: [B, T_keep]
    """
    B, T, D = x.shape
    T_keep = int(T * (1 - mask_ratio))

    noise = torch.rand(B, T, device=x.device)  # [B, T]
    ids_shuffle = torch.argsort(noise, dim=1)  # [B, T]
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, T]

    ids_keep = ids_shuffle[:, :T_keep]  # [B, T_keep]
    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, T, device=x.device)
    mask[:, :T_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # unshuffle to original order
    return x_keep, mask, ids_restore, ids_keep


class ECGMAE_1D(nn.Module):
    """
    Minimal-but-standard MAE for 1D ECG.
    - Encoder sees only visible (kept) tokens.
    - Decoder reconstructs full sequence.
    """

    def __init__(
        self,
        n_leads=1,
        patch_size=16,
        d_model=128,
        enc_depth=4,
        dec_depth=2,
        n_heads=4,
        dim_ff=256,
        dropout=0.1,
        mask_ratio=0.6,
        pos_max_len=4096,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.d_model = d_model

        self.patch_embed = PatchEmbed1D(n_leads, d_model, patch_size)
        self.pos_embed_enc = SinCosPosEmbed(d_model, pos_max_len)

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dim_ff, dropout)
                for _ in range(enc_depth)
            ]
        )

        # Decoder: separate pos embedding and blocks
        self.pos_embed_dec = SinCosPosEmbed(d_model, pos_max_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dim_ff, dropout)
                for _ in range(dec_depth)
            ]
        )

        self.pred_head = nn.Linear(d_model, n_leads * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pred_head.weight, std=0.02)
        nn.init.zeros_(self.pred_head.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, L] -> patches: [B, T, N*P]
        """
        B, N, L = x.shape
        assert N == self.n_leads
        assert L % self.patch_size == 0
        patches = x.unfold(2, self.patch_size, self.patch_size)  # [B, N, T, P]
        patches = patches.permute(0, 2, 1, 3).contiguous()  # [B, T, N, P]
        return patches.view(B, patches.size(1), N * self.patch_size)  # [B, T, N*P]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_patches: [B, T, N*P]
            target_patches: [B, T, N*P]
            mask: [B, T] (1=masked, 0=kept)
        """
        target = self.patchify(x)  # [B, T, N*P]
        tokens = self.patch_embed(x)  # [B, T, D]
        tokens = self.pos_embed_enc(tokens)

        # Masking: encoder only sees kept tokens
        x_keep, mask, ids_restore, ids_keep = random_masking_with_ids(
            tokens, self.mask_ratio
        )

        h = x_keep
        for blk in self.encoder:
            h = blk(h)

        # Prepare decoder tokens: insert mask tokens, then restore original order
        B, T, D = tokens.shape
        T_keep = h.size(1)

        mask_tokens = self.mask_token.expand(B, T - T_keep, D)
        dec_tokens_ = torch.cat([h, mask_tokens], dim=1)  # [B, T, D] but shuffled order
        dec_tokens = torch.gather(
            dec_tokens_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        dec_tokens = self.pos_embed_dec(dec_tokens)
        for blk in self.decoder:
            dec_tokens = blk(dec_tokens)

        pred = self.pred_head(dec_tokens)  # [B, T, N*P]
        return pred, target, mask

    @torch.no_grad()
    def encode(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        Encode ECG into latent representations for downstream tasks.

        Args:
            x: [B, N, L]
            pool: "mean" | "none"
        Returns:
            z:
              - if pool="none": [B, T, D]
              - if pool="mean": [B, D]
        """
        tokens = self.patch_embed(x)  # [B, T, D]
        tokens = self.pos_embed_enc(tokens)

        h = tokens
        for blk in self.encoder:
            h = blk(h)  # [B, T, D]

        if pool == "mean":
            return h.mean(dim=1)  # [B, D]
        elif pool == "none":
            return h  # [B, T, D]
        else:
            raise ValueError(f"Unknown pool={pool}")


def mae_loss_masked(
    pred_patches: torch.Tensor, target_patches: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    mask: [B, T], 1=masked, 0=visible
    """
    # Compute per-patch MSE then average only masked positions
    loss_per = (pred_patches - target_patches).pow(2).mean(dim=-1)  # [B, T]
    loss = (loss_per * mask).sum() / (mask.sum() + 1e-8)
    return loss
