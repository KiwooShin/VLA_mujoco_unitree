"""
code.policy.small_vla.model — GroundedNav student (Architecture A / C).

ADR-001 schema:
  ego_rgb   : (B, C_rgb, 128, 128)   C_rgb=3 or 4 (configurable)
  lang_emb  : (B, 2048)              cached GR00T-LM embedding
  proprio_h : (B, K, 55)             K history frames of proprio
  goal_gt   : (B, 3)                 (dist, cosθ, sinθ) — teacher-forced in train
  vel_gt    : (B, 3)                 (vx, vy, ωz)      — teacher-forced in train
  action    : (B, H, 15)             joint targets, H = chunk horizon

All vision weights are FROM SCRATCH. No pretrained checkpoint is loaded.

RF-1: split out of code/small_vla.py (see code/small_vla.py, the old-path
compat alias, and docs/refactor_plan.md) — this is the top-level-student
half of the model/blocks/heads split. GroundedNav MUST remain importable at
the old `code.small_vla` path: it is constructed fresh by every training/eval
script and then loaded via `model.load_state_dict(ckpt['model_state'])`
(checkpoints store a plain tensor-name-keyed state dict, not a pickled class
reference, so only the *import path* — not any class identity embedded in a
pickle — needs to keep resolving; see code/small_vla.py's compat alias).
"""

from __future__ import annotations
import math
from typing import Any

import torch
import torch.nn as nn

from code.policy.small_vla.blocks import ProprioEncoder, TinyViT
from code.policy.small_vla.heads import ActionHead, DoneHead, GroundingHead, VelocityHead

# ---------------------------------------------------------------------------
# Defaults (overridden by config or constructor kwargs)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = dict(
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

    def __init__(self, arch: str = 'A', teacher_forcing: bool = True, **cfg: Any) -> None:
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
            action_feat_dim = (vis_dim + lang_proj_dim + proprio_enc_dim
                               + C['goal_proj_dim'] + C['vel_proj_dim'])
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
        gt_goal: torch.Tensor | None = None,   # (B, 3) — teacher-forced if arch=A
        gt_vel: torch.Tensor | None = None,    # (B, 3) — teacher-forced if arch=A
    ) -> dict[str, torch.Tensor]:
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
    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        breakdown: dict[str, int] = {}
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
