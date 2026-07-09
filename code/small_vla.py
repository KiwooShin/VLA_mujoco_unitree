"""
GroundedNav student — Architecture A (modular) and C (baseline control).

ADR-001 schema:
  ego_rgb   : (B, C_rgb, 128, 128)   C_rgb=3 or 4 (configurable)
  lang_emb  : (B, 2048)              cached GR00T-LM embedding
  proprio_h : (B, K, 55)             K history frames of proprio
  goal_gt   : (B, 3)                 (dist, cosθ, sinθ) — teacher-forced in train
  vel_gt    : (B, 3)                 (vx, vy, ωz)      — teacher-forced in train
  action    : (B, H, 15)             joint targets, H = chunk horizon

All vision weights are FROM SCRATCH. No pretrained checkpoint is loaded.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defaults (overridden by config or constructor kwargs)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    img_size=128,
    in_ch=3,           # 3=RGB, 4=RGBD
    patch_size=8,      # TinyViT patch
    vit_depth=4,
    vit_heads=4,
    vit_dim=128,       # token dim
    vit_ff_mult=2,
    lang_dim=2048,     # GR00T-LM embedding dim
    proprio_dim=55,    # 30 (q,qd) + 10 (IMU) + 15 (prev_action)
    proprio_K=6,       # history length
    gru_hidden=128,
    goal_dim=3,        # (dist, cosθ, sinθ)
    vel_dim=3,         # (vx, vy, ωz)
    action_dim=15,     # joint targets
    chunk_H=1,         # action chunking horizon (set to 16 for demo)
    lang_proj_dim=128, # projected lang embedding dim
    goal_proj_dim=64,
    vel_proj_dim=64,
    dropout=0.1,
    vel_proprio=False, # V6: if True, vel head also takes (proprio_emb, gait-phase[sin,cos])
)


# ---------------------------------------------------------------------------
# TinyViT — small from-scratch patch transformer
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Non-overlapping patch tokenizer."""
    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int):
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
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
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
    def __init__(self, dim: int, heads: int, ff_mult: int = 2, dropout: float = 0.0):
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
                 dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, proprio_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(proprio_dim, hidden, batch_first=True, num_layers=1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, proprio_dim) → (B, hidden)."""
        _, h = self.gru(x)          # h: (1, B, hidden)
        return self.drop(h.squeeze(0))


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
                 n_patches: int = 256):
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
                 vel_proprio: bool = False, proprio_enc_dim: int = 128):
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
                proprio_emb: Optional[torch.Tensor] = None,
                phase: Optional[torch.Tensor] = None) -> torch.Tensor:
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
                 n_dec_layers: int = 2, n_heads: int = 4, dropout: float = 0.0):
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
    def __init__(self, vis_dim: int, lang_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vis_dim + lang_dim, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, vis: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([vis, lang], dim=-1)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# GroundedNav — top-level student
# ---------------------------------------------------------------------------

class GroundedNav(nn.Module):
    """
    Arch A: full modular (grounding → vel → action + done).
    Arch C: baseline control (no grounding/vel heads; action on vis+lang+proprio).

    Args:
        arch: 'A' or 'C'
        teacher_forcing: if True, forward() accepts gt_goal / gt_vel and uses
                         them in downstream heads instead of predicted values.
    """

    def __init__(self, arch: str = 'A', teacher_forcing: bool = True, **cfg):
        super().__init__()
        C = {**DEFAULTS, **cfg}
        self.arch = arch.upper()
        self.teacher_forcing = teacher_forcing
        self.chunk_H = C['chunk_H']

        # --- Vision backbone (from scratch) ---
        self.vision = TinyViT(
            img_size=C['img_size'],
            patch_size=C['patch_size'],
            in_ch=C['in_ch'],
            dim=C['vit_dim'],
            depth=C['vit_depth'],
            heads=C['vit_heads'],
            ff_mult=C['vit_ff_mult'],
            dropout=C['dropout'],
        )
        vis_dim = C['vit_dim']

        # --- Language projection (lang_emb is input, not computed here) ---
        self.lang_proj = nn.Sequential(
            nn.Linear(C['lang_dim'], C['lang_proj_dim']),
            nn.LayerNorm(C['lang_proj_dim']),
            nn.GELU(),
        )
        lang_proj_dim = C['lang_proj_dim']

        # --- Proprio GRU ---
        self.proprio_enc = ProprioEncoder(C['proprio_dim'], C['gru_hidden'], C['dropout'])
        proprio_enc_dim = C['gru_hidden']

        # --- Arch A: grounding + velocity heads ---
        self.vel_proprio = C.get('vel_proprio', False)  # V6: proprio-fed vel head flag
        if self.arch == 'A':
            n_patches = (C['img_size'] // C['patch_size']) ** 2  # 16x16 = 256
            self.grounding = GroundingHead(vis_dim, lang_proj_dim, C['goal_dim'],
                                           n_patches=n_patches)
            # V6: optionally feed proprio_emb + phase to vel head
            self.velocity = VelocityHead(
                C['goal_dim'], vis_dim, lang_proj_dim, C['vel_dim'],
                vel_proprio=self.vel_proprio,
                proprio_enc_dim=proprio_enc_dim,
            )
            self.goal_proj = nn.Sequential(
                nn.Linear(C['goal_dim'], C['goal_proj_dim']), nn.GELU()
            )
            self.vel_proj = nn.Sequential(
                nn.Linear(C['vel_dim'], C['vel_proj_dim']), nn.GELU()
            )
            action_feat_dim = vis_dim + lang_proj_dim + proprio_enc_dim + C['goal_proj_dim'] + C['vel_proj_dim']
        else:  # Arch C
            action_feat_dim = vis_dim + lang_proj_dim + proprio_enc_dim

        # --- Action head ---
        # Transformer decoder, feature dim must match n_heads divisibility
        # Round up to next multiple of vit_heads
        ah_heads = C['vit_heads']
        if action_feat_dim % ah_heads != 0:
            action_feat_dim = math.ceil(action_feat_dim / ah_heads) * ah_heads
        self.action_feat_dim = action_feat_dim

        self.action_feat_proj = nn.Linear(
            (vis_dim + lang_proj_dim + proprio_enc_dim +
             (C['goal_proj_dim'] + C['vel_proj_dim'] if self.arch == 'A' else 0)),
            action_feat_dim,
        )

        self.action_head = ActionHead(
            feat_dim=action_feat_dim,
            action_dim=C['action_dim'],
            chunk_H=self.chunk_H,
            n_dec_layers=2,
            n_heads=ah_heads,
            dropout=C['dropout'],
        )

        # --- Done head ---
        self.done_head = DoneHead(vis_dim, lang_proj_dim)

        self._C = C

    # ------------------------------------------------------------------
    def forward(
        self,
        ego_rgb: torch.Tensor,          # (B, in_ch, 128, 128)
        lang_emb: torch.Tensor,         # (B, 2048)
        proprio_h: torch.Tensor,        # (B, K, 55)
        gt_goal: Optional[torch.Tensor] = None,   # (B, 3) — teacher-forced if arch=A
        gt_vel: Optional[torch.Tensor] = None,    # (B, 3) — teacher-forced if arch=A
    ) -> dict:
        """
        Returns a dict with keys present depending on arch:
          'action'   : (B, H, 15)
          'done'     : (B,)   logit
          'goal'     : (B, 3) predicted (arch A only)
          'vel'      : (B, 3) predicted (arch A only)
        """
        # Vision
        vis_patches, vis_pooled = self.vision(ego_rgb)   # (B, N, vit_dim), (B, vit_dim)

        # Language
        lang = self.lang_proj(lang_emb)                # (B, lang_proj_dim)

        # Proprio
        prop = self.proprio_enc(proprio_h)             # (B, gru_hidden)

        out = {}

        if self.arch == 'A':
            # Grounding: use patch tokens for spatial localization
            goal_pred = self.grounding(vis_patches, lang)  # (B, 3)
            out['goal'] = goal_pred

            # V6: for proprio-fed vel head, extract gait-phase from the last proprio frame.
            # proprio_h shape: (B, K, proprio_dim). When proprio_dim>=57, the last 2 dims
            # of each frame are [sin(phi), cos(phi)]. Use the most recent frame (t-1).
            if self.vel_proprio:
                # phase from most recent proprio frame: last 2 cols
                phase_feat = proprio_h[:, -1, -2:]  # (B, 2)
                vel_pred = self.velocity(goal_pred, vis_pooled, lang, prop, phase_feat)
            else:
                vel_pred = self.velocity(goal_pred, vis_pooled, lang)
            out['vel'] = vel_pred

            # Teacher forcing
            if self.teacher_forcing and gt_goal is not None:
                goal_in = gt_goal
            else:
                goal_in = goal_pred
            if self.teacher_forcing and gt_vel is not None:
                vel_in = gt_vel
            else:
                vel_in = vel_pred

            goal_emb = self.goal_proj(goal_in)   # (B, goal_proj_dim)
            vel_emb = self.vel_proj(vel_in)       # (B, vel_proj_dim)

            feat_raw = torch.cat([vis_pooled, lang, prop, goal_emb, vel_emb], dim=-1)
        else:
            feat_raw = torch.cat([vis_pooled, lang, prop], dim=-1)

        # Action
        feat = self.action_feat_proj(feat_raw)         # (B, action_feat_dim)
        actions = self.action_head(feat)               # (B, H, action_dim)
        out['action'] = actions

        # Done
        done_logit = self.done_head(vis_pooled, lang)  # (B,)
        out['done'] = done_logit

        return out

    # ------------------------------------------------------------------
    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        breakdown = {}
        for name, mod in [
            ('vision', self.vision),
            ('lang_proj', self.lang_proj),
            ('proprio_enc', self.proprio_enc),
            ('action_head', self.action_head),
            ('done_head', self.done_head),
        ]:
            breakdown[name] = sum(p.numel() for p in mod.parameters())
        if self.arch == 'A':
            breakdown['grounding'] = sum(p.numel() for p in self.grounding.parameters())
            breakdown['velocity'] = sum(p.numel() for p in self.velocity.parameters())
        breakdown['total'] = total
        return breakdown
