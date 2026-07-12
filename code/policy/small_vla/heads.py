"""
code.policy.small_vla.heads — GroundedNav output heads.

Arch A's grounding/velocity heads plus the shared action-chunking and done
heads:
  - GroundingHead : column-attention bearing/distance estimator (Arch A only).
  - VelocityHead  : (goal, vis, lang[, proprio, phase]) -> (vx, vy, ωz).
  - ActionHead    : chunked joint-target decoder (Arch A and C).
  - DoneHead      : (vis, lang) -> done logit.

RF-1: split out of code/small_vla.py (see code/small_vla.py, the old-path
compat alias, and docs/refactor_plan.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grounding head (Arch A only)
# ---------------------------------------------------------------------------

class GroundingHead(nn.Module):
    """
    Column-attention grounding head: explicitly uses the horizontal position of
    patches to estimate bearing.

    Physics: target bearing ≈ f(column of target in image). If target is in
    column c (0=left, W-1=right), bearing ≈ atan2(c - W/2, focal_length).

    Implementation:
      1. Compute per-patch object-presence score via (vis_patch · lang_key).
      2. Reshape to (B, H, W) grid; sum over rows → column weights (B, W).
      3. Apply softmax → column attention p(c). Compute expected column = Σ c·p(c).
      4. Map expected column to (cosθ, sinθ) via a learned calibration MLP.
      5. Distance estimated from attended visual features via separate MLP.

    This directly encodes the physical relationship between image column and bearing,
    making the learning problem much easier.
    """
    def __init__(self, vis_dim: int, lang_dim: int, goal_dim: int = 3,
                 n_patches: int = 256) -> None:
        super().__init__()
        self.vis_dim    = vis_dim
        self.goal_dim   = goal_dim
        self.n_patches  = n_patches  # 16×16 = 256 for 128px / 8px patches
        self.patch_grid = int(n_patches ** 0.5)  # 16 (W = H = 16 columns/rows)

        # Language → key/query vectors for dot-product attention
        self.lang_key = nn.Linear(lang_dim, vis_dim)

        # Per-patch scoring: project patch features for dot-product scoring
        self.patch_proj = nn.Linear(vis_dim, vis_dim)

        # Column → (cosθ, sinθ): maps normalized column position to bearing
        # Input: (expected_col_normalized, col_softmax_entropy)
        self.bearing_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(),
            nn.Linear(32, 2),   # → (cosθ, sinθ), unnormalized
        )

        # Distance estimator: attended patch features → dist
        self.dist_mlp = nn.Sequential(
            nn.Linear(vis_dim + lang_dim, 64), nn.GELU(),
            nn.Linear(64, 1),   # → scalar dist
        )

        # Fallback MLP for when vis is CLS-pooled (B,D) instead of patches (B,N,D)
        in_dim = vis_dim + lang_dim
        self.cls_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, goal_dim),
        )

    def forward(self, vis: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """
        vis : (B, N, D) patch tokens  OR  (B, D) CLS-pooled token
        lang: (B, lang_dim)
        Returns (B, 3): [dist, cosθ, sinθ] (raw, normalized at loss time).
        """
        if vis.dim() == 2:
            # CLS-pooled fallback (no spatial info): use old MLP path
            return self.cls_net(torch.cat([vis, lang], dim=-1))

        B, N, D = vis.shape
        G = self.patch_grid  # 16

        # --- Step 1: Compute per-patch presence score ---
        # patch_proj: (B, N, D)
        patches_proj = self.patch_proj(vis)   # (B, N, D)
        lang_q = self.lang_key(lang)          # (B, D)
        # Dot-product score: (B, N)
        scores = (patches_proj * lang_q.unsqueeze(1)).sum(-1)  # (B, N)
        # scores = scores / (D ** 0.5)   # scale by sqrt(D)

        # --- Step 2: Reshape to 2D grid and compute column marginal ---
        # patches are in row-major order: patch i → (row=i//G, col=i%G)
        scores_2d = scores.view(B, G, G)     # (B, rows, cols)
        col_logits = scores_2d.sum(dim=1)    # (B, cols) — sum over rows

        # --- Step 3: Softmax → column probability distribution ---
        col_probs = F.softmax(col_logits, dim=-1)   # (B, G)

        # Column indices normalized to [-1, 1].
        # MuJoCo camera convention: col 0 = world-RIGHT (positive bearing direction),
        # col G-1 = world-LEFT (negative bearing direction).
        # So col_idx goes from +1 → -1 (right-to-left in world frame).
        col_idx = torch.linspace(1.0, -1.0, G, device=vis.device)  # (G,)
        expected_col = (col_probs * col_idx).sum(-1, keepdim=True)  # (B, 1) ∈ [-1, 1]

        # Entropy of column distribution (uncertainty in localization)
        entropy = -(col_probs * (col_probs + 1e-8).log()).sum(-1, keepdim=True)  # (B, 1)
        entropy = entropy / math.log(G)  # normalize to [0, 1]

        # --- Step 4: Bearing from column position ---
        bear_feat = torch.cat([expected_col, entropy], dim=-1)  # (B, 2)
        bearing = self.bearing_mlp(bear_feat)  # (B, 2): [cosθ, sinθ] unnormalized

        # --- Step 5: Distance from attended patch + lang ---
        # Attend patches by col_probs (expand to full N = G*G patches)
        patch_attn = col_probs.unsqueeze(1).expand(-1, G, -1)  # (B, G, G)
        patch_attn = patch_attn.reshape(B, N)                  # (B, N)
        attended = (vis * patch_attn.unsqueeze(-1)).sum(1)      # (B, D)
        dist_feat = torch.cat([attended, lang], dim=-1)         # (B, D + lang_dim)
        dist = self.dist_mlp(dist_feat)                         # (B, 1)

        return torch.cat([dist, bearing], dim=-1)  # (B, 3)


# ---------------------------------------------------------------------------
# Velocity head (Arch A only)
# ---------------------------------------------------------------------------

class VelocityHead(nn.Module):
    """
    (goal, vis, lang[, proprio_emb, phase]) → (vx, vy, ωz).

    When vel_proprio=True, also accepts:
      proprio_emb : (B, proprio_enc_dim)  — GRU hidden state of proprio encoder
      phase       : (B, 2)               — [sin(phi), cos(phi)] gait phase

    The old behavior (vel_proprio=False) is preserved: forward() still accepts
    (goal, vis, lang) with no extra args.
    """
    def __init__(self, goal_dim: int, vis_dim: int, lang_dim: int, vel_dim: int = 3,
                 vel_proprio: bool = False, proprio_enc_dim: int = 128) -> None:
        super().__init__()
        self.vel_proprio = vel_proprio
        if vel_proprio:
            # Adds: proprio_enc_dim + 2 (phase)
            in_dim = goal_dim + vis_dim + lang_dim + proprio_enc_dim + 2
        else:
            in_dim = goal_dim + vis_dim + lang_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, vel_dim),
        )

    def forward(self, goal: torch.Tensor, vis: torch.Tensor, lang: torch.Tensor,
                proprio_emb: torch.Tensor | None = None,
                phase: torch.Tensor | None = None) -> torch.Tensor:
        if self.vel_proprio and proprio_emb is not None and phase is not None:
            return self.net(torch.cat([goal, vis, lang, proprio_emb, phase], dim=-1))
        return self.net(torch.cat([goal, vis, lang], dim=-1))


# ---------------------------------------------------------------------------
# Action head — supports chunking H
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """
    (feat_cat, chunk_H) → (B, H, action_dim).
    feat_cat = [vis_pooled | lang_proj | proprio_enc | vel_proj | goal_proj]
    The head uses a small transformer decoder to produce H action tokens in parallel
    (like ACT / Diffusion Policy's simple chunking mode), which is compatible with
    temporal ensembling at deploy time.
    """
    def __init__(self, feat_dim: int, action_dim: int, chunk_H: int,
                 n_dec_layers: int = 2, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.H = chunk_H
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # Project input feat to transformer dim
        self.in_proj = nn.Linear(feat_dim, feat_dim)
        # Learned action query tokens (H queries)
        self.queries = nn.Parameter(torch.zeros(1, chunk_H, feat_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=n_heads,
            dim_feedforward=feat_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.out_proj = nn.Linear(feat_dim, action_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, feat_dim) → actions: (B, H, action_dim)."""
        B = feat.shape[0]
        memory = self.in_proj(feat).unsqueeze(1)       # (B, 1, feat_dim) as memory
        queries = self.queries.expand(B, -1, -1)        # (B, H, feat_dim)
        out = self.decoder(queries, memory)             # (B, H, feat_dim)
        return self.out_proj(out)                       # (B, H, action_dim)


# ---------------------------------------------------------------------------
# Done head
# ---------------------------------------------------------------------------

class DoneHead(nn.Module):
    """(vis, lang) → done logit (BCE)."""
    def __init__(self, vis_dim: int, lang_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vis_dim + lang_dim, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, vis: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([vis, lang], dim=-1)).squeeze(-1)  # (B,)
