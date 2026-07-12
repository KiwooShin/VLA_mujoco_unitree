"""
code.policy.small_vla.blocks — from-scratch vision/proprio encoder blocks.

Building blocks shared by both GroundedNav architectures (A and C):
  - PatchEmbed, Attention, TransformerBlock, TinyViT : vision backbone.
  - ProprioEncoder                                    : proprio GRU encoder.

All weights are FROM SCRATCH. No pretrained checkpoint is loaded.

RF-1: split out of code/small_vla.py (see code/small_vla.py, the old-path
compat alias, and docs/refactor_plan.md) — this is the generic-block half
of the model/blocks/heads split; code.policy.small_vla.heads holds the
grounding/velocity/action/done heads, code.policy.small_vla.model holds the
top-level GroundedNav student (which MUST remain importable at the old
`code.small_vla` path — see that module's docstring).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TinyViT — small from-scratch patch transformer
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Non-overlapping patch tokenizer."""
    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W) → (B,N,D)
        x = self.proj(x)          # (B, D, h, w)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention block."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + MLP with residual connections."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """From-scratch vision encoder: ego RGBD → patch tokens + pooled feature."""
    def __init__(self, img_size: int, patch_size: int, in_ch: int,
                 dim: int, depth: int, heads: int, ff_mult: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, ff_mult, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch_tokens: B×N×D, pooled: B×D)."""
        B = x.shape[0]
        tok = self.patch_embed(x)                        # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tok = torch.cat([cls, tok], dim=1)               # (B, 1+N, D)
        tok = tok + self.pos_embed
        tok = self.blocks(tok)
        tok = self.norm(tok)
        return tok[:, 1:], tok[:, 0]                     # patch_tokens, cls_pooled


# ---------------------------------------------------------------------------
# Proprio GRU encoder
# ---------------------------------------------------------------------------

class ProprioEncoder(nn.Module):
    """GRU over K proprio frames → hidden state."""
    def __init__(self, proprio_dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(proprio_dim, hidden, batch_first=True, num_layers=1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, proprio_dim) → (B, hidden)."""
        _, h = self.gru(x)          # h: (1, B, hidden)
        return self.drop(h.squeeze(0))
