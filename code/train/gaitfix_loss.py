"""code.train.gaitfix_loss — GaitFix (Fix 1) residual/standardized action loss.

Split out of the original ``train_gaitfix.py`` (RF-1): this module owns the
loss definition only (``GaitFixLoss`` + the joint-name table used for
logging). The per-epoch training loop lives in ``code.train.gaitfix_epoch``;
the CLI entry point lives in ``code.train.gaitfix``.

Fix 1 Implementation
--------------------
Instead of predicting absolute joint angles (which mode-averages to the mean pose), the
model predicts PER-JOINT STANDARDIZED DELTAS from the default pose:

    delta_j       = action_j - default_angles_j
    normed_delta_j = (delta_j - mean_j) / std_j     [computed over train set]

    Model output  = normed_delta                     [standardized space]
    Loss          = smooth-L1(pred_normed, gt_normed) in standardized space
    Deploy        = default_angles + (pred_normed * std + mean)   → absolute target_dof

Higher-variance swing joints (knee, hip_pitch) naturally get learned first since they have
more signal. Optional swing-joint up-weighting is supported (--swing-weight).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Joint names (for logging)
# ---------------------------------------------------------------------------
JOINT_NAMES: list[str] = [
    'l_hip_pitch', 'l_hip_roll',  'l_hip_yaw',
    'l_knee',      'l_ank_pitch', 'l_ank_roll',
    'r_hip_pitch', 'r_hip_roll',  'r_hip_yaw',
    'r_knee',      'r_ank_pitch', 'r_ank_roll',
    'waist_yaw',   'waist_roll',  'waist_pitch',
]


# ---------------------------------------------------------------------------
# GaitFix Loss (standardized residual space)
# ---------------------------------------------------------------------------

class GaitFixLoss(nn.Module):
    """Multi-task loss with residual-standardized action target (Fix 1).

    The model is trained to predict standardized deltas:
        normed_delta = (action - default_angles - mean) / std

    The action loss is computed in standardized space (unit-variance per joint).
    Optionally, higher-variance swing joints are up-weighted.

    Other heads (goal, vel, done) use the same smooth-L1 as before.

    Args:
        action_stats: Dict with 'mean' (15,), 'std' (15,), and
            'default_angles' (15,) arrays used to standardize actions.
        huber_beta: Beta (transition point) for the smooth-L1 losses.
        w_action: Weight for the action loss term.
        w_goal: Weight for the goal loss term.
        w_vel: Weight for the velocity loss term.
        w_done: Weight for the done (BCE) loss term.
        swing_weight: Multiplier applied to the top-5 highest-variance
            (swing) joints in the action loss.
        device: Device string the stat buffers are placed on.
    """

    def __init__(
        self,
        action_stats: dict,                # {'mean': (15,), 'std': (15,), 'default_angles': (15,)}
        huber_beta: float = 0.1,
        w_action: float = 5.0,
        w_goal:   float = 1.0,
        w_vel:    float = 1.0,
        w_done:   float = 1.0,
        swing_weight: float = 1.0,         # multiplier for top-5 highest-variance joints
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.beta     = huber_beta
        self.w_action = w_action
        self.w_goal   = w_goal
        self.w_vel    = w_vel
        self.w_done   = w_done

        # Register stats as buffers (moved to device with .to())
        dev = torch.device(device)
        mean  = torch.from_numpy(np.array(action_stats['mean'],           dtype=np.float32)).to(dev)
        std   = torch.from_numpy(np.array(action_stats['std'],            dtype=np.float32)).to(dev)
        deflt = torch.from_numpy(np.array(action_stats['default_angles'], dtype=np.float32)).to(dev)
        self.register_buffer('_mean',  mean)
        self.register_buffer('_std',   std)
        self.register_buffer('_deflt', deflt)

        # Per-joint loss weights (1 + extra for swing joints)
        joint_w = torch.ones(15, dtype=torch.float32, device=dev)
        if swing_weight > 1.0:
            # Up-weight top-5 highest-variance joints
            std_np    = np.array(action_stats['std'], dtype=np.float32)
            swing_idx = np.argsort(std_np)[::-1][:5]
            for idx in swing_idx:
                joint_w[int(idx)] = float(swing_weight)
        self.register_buffer('_joint_w', joint_w)  # (15,)

    def normalize_action(self, action_abs: torch.Tensor) -> torch.Tensor:
        """Converts absolute action (B, H, 15) to standardized delta (B, H, 15).

        Args:
            action_abs: Absolute joint-angle action tensor, shape (B, H, 15).

        Returns:
            Standardized delta tensor, shape (B, H, 15).
        """
        delta = action_abs - self._deflt                       # (B, H, 15)
        return (delta - self._mean) / self._std               # normalized

    def denormalize_action(self, normed: torch.Tensor) -> torch.Tensor:
        """Converts standardized delta (*, 15) to absolute action (*, 15).

        Args:
            normed: Standardized delta tensor, any leading shape with a
                trailing 15-dim.

        Returns:
            Absolute joint-angle action tensor, same shape as ``normed``.
        """
        return self._deflt + normed * self._std + self._mean

    def _huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes mean smooth-L1 loss between pred and target."""
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction='mean')

    def _huber_weighted(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-joint weighted smooth-L1. pred/target: (B, H, 15) or (B, 15)."""
        per_elem = F.smooth_l1_loss(pred, target, beta=self.beta, reduction='none')  # same shape
        # Apply per-joint weight (last dim = 15)
        per_elem = per_elem * self._joint_w                      # broadcast over B, H
        return per_elem.mean()

    def forward(
        self,
        preds: dict,
        *,
        action_gt_abs: torch.Tensor,         # (B, H, 15) absolute joint angles
        done_gt:       torch.Tensor,          # (B,) float 0/1
        goal_gt:       torch.Tensor | None = None,   # (B, 3)
        vel_gt:        torch.Tensor | None = None,   # (B, 3)
    ) -> tuple[torch.Tensor, dict]:
        """Computes the total multi-task loss and a per-term breakdown.

        ``preds['action']`` is in STANDARDIZED DELTA SPACE (model output).
        ``action_gt_abs`` is in ABSOLUTE space; it is normalized here before
        computing the action loss.

        Args:
            preds: Model output dict, expected to contain 'action', 'done',
                and optionally 'goal'/'vel' tensors.
            action_gt_abs: Ground-truth absolute joint angles, (B, H, 15).
            done_gt: Ground-truth done flags, (B,) float 0/1.
            goal_gt: Optional ground-truth goal, (B, 3).
            vel_gt: Optional ground-truth velocity command, (B, 3).

        Returns:
            A tuple of (total weighted loss tensor, dict of per-term float
            loss values including 'total').
        """
        losses = {}
        dev   = action_gt_abs.device
        total = torch.tensor(0.0, device=dev, dtype=action_gt_abs.dtype)

        # ---- Action loss (standardized space) ----
        pred_normed = preds['action']                            # (B, H, 15)
        if action_gt_abs.dim() == 2:
            action_gt_abs = action_gt_abs.unsqueeze(1)           # (B, 1, 15)
        gt_normed   = self.normalize_action(action_gt_abs)       # (B, H, 15)
        # Expand if needed
        if gt_normed.shape[1] != pred_normed.shape[1]:
            gt_normed = gt_normed.expand_as(pred_normed)

        l_action = self._huber_weighted(pred_normed, gt_normed)
        losses['action'] = l_action.item()
        total = total + self.w_action * l_action

        # ---- Done (BCE) ----
        l_done = F.binary_cross_entropy_with_logits(preds['done'], done_gt)
        losses['done'] = l_done.item()
        total = total + self.w_done * l_done

        # ---- Goal (Arch A only) ----
        if 'goal' in preds and goal_gt is not None:
            l_goal = self._huber(preds['goal'], goal_gt)
            losses['goal'] = l_goal.item()
            total = total + self.w_goal * l_goal
        else:
            losses['goal'] = 0.0

        # ---- Velocity (Arch A only) ----
        if 'vel' in preds and vel_gt is not None:
            l_vel = self._huber(preds['vel'], vel_gt)
            losses['vel'] = l_vel.item()
            total = total + self.w_vel * l_vel
        else:
            losses['vel'] = 0.0

        losses['total'] = total.item()
        return total, losses
