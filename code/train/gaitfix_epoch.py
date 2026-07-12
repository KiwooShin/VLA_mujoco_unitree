"""code.train.gaitfix_epoch — shared train/val epoch runner + velocity-head audit.

Split out of the original ``train_gaitfix.py`` (RF-1). ``_run_epoch`` is reused
by ``code.train.gaitfix`` (Fix 1), ``code.train.dart_phase`` (Fix 4+5), and
``code.train.maneuver`` (proprio_dim=62 fine-tune) — all three build a
``GroundedNav`` + ``GaitFixLoss`` pair and share this one epoch loop.
``audit_velocity_head`` is the Fix-2 diagnostic (vel head near-zero check).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from code.small_vla import GroundedNav
from code.train.gaitfix_loss import GaitFixLoss

# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model:     GroundedNav,
    loader:    DataLoader,
    loss_fn:   GaitFixLoss,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
    verbose:   bool = False,
) -> dict:
    """Runs one training or validation epoch over ``loader``.

    Vision input is zeroed out (not the stability bottleneck; keeps
    iteration fast). When ``train`` is True and ``optimizer`` is given,
    performs backprop and a gradient-clipped optimizer step per batch.

    Args:
        model: The GroundedNav model to run forward (and optionally backward)
            through.
        loader: DataLoader yielding batches with 'ego_rgb', 'lang_emb',
            'proprio_h', 'action', 'goal', 'vel_cmd', and 'done' keys.
        loss_fn: The GaitFixLoss instance used to compute the loss.
        optimizer: Optimizer to step when training; ignored/unused when
            ``train`` is False.
        device: Torch device to move batch tensors to.
        train: Whether to run in training mode (grad enabled, backprop) or
            eval mode (no grad).
        verbose: Unused; reserved for future per-batch logging.

    Returns:
        A dict of epoch-averaged metrics: 'total', 'action', 'done', 'goal',
        'vel', and 'done_acc'.
    """
    model.train(train)
    total_loss = action_loss = done_loss = goal_loss = vel_loss = 0.0
    n_done_correct = n_total = n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            ego_rgb   = batch['ego_rgb'].to(device)
            lang_emb  = batch['lang_emb'].to(device)
            proprio_h = batch['proprio_h'].to(device)
            action_gt = batch['action'].to(device)     # ABSOLUTE joint angles
            goal_gt   = batch['goal'].to(device)
            vel_gt    = batch['vel_cmd'].to(device)
            done_gt   = batch['done'].to(device)

            if train and optimizer is not None:
                optimizer.zero_grad()

            # Vision: zero out (not the stability issue; keeps iteration fast)
            ego_rgb_zeros = torch.zeros_like(ego_rgb)

            preds = model(
                ego_rgb_zeros, lang_emb, proprio_h,
                gt_goal=goal_gt, gt_vel=vel_gt,
            )
            # preds['action'] = raw model output in STANDARDIZED DELTA SPACE
            loss, breakdown = loss_fn(
                preds,
                action_gt_abs=action_gt,
                done_gt=done_gt,
                goal_gt=goal_gt,
                vel_gt=vel_gt,
            )

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss  += breakdown['total']
            action_loss += breakdown['action']
            done_loss   += breakdown['done']
            goal_loss   += breakdown['goal']
            vel_loss    += breakdown['vel']

            done_pred = (preds['done'].detach() > 0).float()
            n_done_correct += (done_pred == done_gt).sum().item()
            n_total        += done_gt.shape[0]
            n_batches      += 1

    nb = max(n_batches, 1)
    return {
        'total':    total_loss  / nb,
        'action':   action_loss / nb,
        'done':     done_loss   / nb,
        'goal':     goal_loss   / nb,
        'vel':      vel_loss    / nb,
        'done_acc': n_done_correct / max(n_total, 1),
    }


# ---------------------------------------------------------------------------
# Velocity head audit (Fix 2 — diagnostic)
# ---------------------------------------------------------------------------

def audit_velocity_head(
    model:        GroundedNav,
    val_loader:   DataLoader,
    device:       torch.device,
    n_batches:    int = 20,
) -> dict:
    """Evaluates the velocity head on val data: compares pred (vx,vy,wz) vs GT.

    Args:
        model: The GroundedNav model to evaluate (vision is zeroed out).
        val_loader: DataLoader over the validation split.
        device: Torch device to move batch tensors to.
        n_batches: Number of batches to evaluate over.

    Returns:
        A dict of pred/GT mean/std/MAE stats per axis (vx, vy, wz), plus
        'n_samples' and 'vel_head_near_zero'. Returns
        ``{'error': ...}`` if the model has no 'vel' head (e.g. arch C).
    """
    model.eval()
    pred_vels = []
    gt_vels   = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            ego_rgb   = torch.zeros_like(batch['ego_rgb']).to(device)
            lang_emb  = batch['lang_emb'].to(device)
            proprio_h = batch['proprio_h'].to(device)
            goal_gt   = batch['goal'].to(device)
            vel_gt    = batch['vel_cmd'].to(device)

            preds = model(ego_rgb, lang_emb, proprio_h, gt_goal=goal_gt)
            if 'vel' in preds:
                pred_vels.append(preds['vel'].cpu().numpy())
            gt_vels.append(vel_gt.cpu().numpy())

    if not pred_vels:
        return {'error': 'No velocity predictions (arch C?)'}

    pred_arr = np.concatenate(pred_vels, axis=0)   # (N, 3)
    gt_arr   = np.concatenate(gt_vels,   axis=0)   # (N, 3)

    stats = {
        'pred_mean_vx':  float(pred_arr[:, 0].mean()),
        'pred_mean_vy':  float(pred_arr[:, 1].mean()),
        'pred_mean_wz':  float(pred_arr[:, 2].mean()),
        'pred_std_vx':   float(pred_arr[:, 0].std()),
        'pred_std_vy':   float(pred_arr[:, 1].std()),
        'pred_std_wz':   float(pred_arr[:, 2].std()),
        'gt_mean_vx':    float(gt_arr[:, 0].mean()),
        'gt_mean_vy':    float(gt_arr[:, 1].mean()),
        'gt_mean_wz':    float(gt_arr[:, 2].mean()),
        'gt_std_vx':     float(gt_arr[:, 0].std()),
        'gt_std_vy':     float(gt_arr[:, 1].std()),
        'gt_std_wz':     float(gt_arr[:, 2].std()),
        'mae_vx':        float(np.abs(pred_arr[:, 0] - gt_arr[:, 0]).mean()),
        'mae_vy':        float(np.abs(pred_arr[:, 1] - gt_arr[:, 1]).mean()),
        'mae_wz':        float(np.abs(pred_arr[:, 2] - gt_arr[:, 2]).mean()),
        'n_samples':     int(pred_arr.shape[0]),
    }
    # Is vel head ~0?  Check if pred mean vx << gt mean vx
    stats['vel_head_near_zero'] = (
        abs(stats['pred_mean_vx']) < 0.05 and
        stats['pred_std_vx'] < 0.10
    )
    return stats
