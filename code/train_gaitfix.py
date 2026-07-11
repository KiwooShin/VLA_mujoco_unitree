"""
train_gaitfix.py — Gait-fix training run: Fix 1 (residual + standardized action target).

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

Architecture: Arch A, vision=ZEROS (zeros fed as RGB — not the stability issue; keeps
training fast). Runs on the 80-episode easy dataset.

Usage
-----
# Overfit gate first (MANDATORY before full training):
MUJOCO_GL=egl python code/train_gaitfix.py --overfit --arch A \
    --data dataset/easy_train80 --out runs/gaitfix_A --overfit-samples 32

# Full training (~15-25 epochs, per-epoch checkpoints):
MUJOCO_GL=egl python code/train_gaitfix.py --arch A \
    --data dataset/easy_train80 --out runs/gaitfix_A --epochs 20 --batch 64 \
    --lr 3e-4 --swing-weight 2.0

# Resume from a specific epoch checkpoint:
MUJOCO_GL=egl python code/train_gaitfix.py --arch A \
    --data dataset/easy_train80 --out runs/gaitfix_A --epochs 20 --resume-ckpt runs/gaitfix_A/epoch_0005.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.action_stats import compute_action_stats, DEFAULT_ANGLES, STD_FLOOR
from code.dataset import make_dataloader, ParquetDataset
from code.small_vla import GroundedNav, DEFAULTS

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
# Overfit gate (real data subset)
# ---------------------------------------------------------------------------

def run_overfit_gate(
    arch:          str,
    device:        torch.device,
    repo_path:     str,
    action_stats:  dict,
    batch_size:    int = 16,
    overfit_n:     int = 32,
    max_epochs:    int = 300,
    target_loss:   float = 0.10,
    lr:            float = 3e-4,
    swing_weight:  float = 2.0,
    verbose:       bool  = True,
) -> dict:
    """Overfits a small fixed subset of real data.

    PASS = action_loss drops below ``target_loss`` within ``max_epochs``.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: Path to the dataset repo (e.g. easy_train80).
        action_stats: Action normalization statistics (mean/std/etc.).
        batch_size: Batch size for the overfit subset loader.
        overfit_n: Number of samples to overfit on.
        max_epochs: Maximum number of overfit epochs before declaring FAIL.
        target_loss: Action loss threshold to declare PASS.
        lr: Learning rate for the AdamW optimizer.
        swing_weight: Swing-joint upweighting factor for GaitFixLoss.
        verbose: Whether to print periodic progress lines.

    Returns:
        A dict with keys 'status' ('PASS' or 'FAIL'), 'epoch', 'action_loss',
        and 'elapsed' (seconds).
    """
    print(f"\n{'='*60}")
    print(f"  GAITFIX OVERFIT GATE — Arch {arch}  swing_weight={swing_weight}")
    print(f"  repo={repo_path}  n_samples={overfit_n}  target_loss={target_loss}")
    print(f"{'='*60}")

    # Build dataset and take a small subset
    full_ds = ParquetDataset(
        repo_path=repo_path, split='train', train_fraction=0.9,
        load_video=False,   # zeros for vision
    )
    n = min(overfit_n, len(full_ds))
    subset = Subset(full_ds, list(range(n)))
    loader = DataLoader(subset, batch_size=min(batch_size, n), shuffle=False, num_workers=0)
    print(f"  Overfit subset: {n} samples from {len(full_ds)} total train frames")

    model = GroundedNav(arch=arch, teacher_forcing=True, chunk_H=1).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = GaitFixLoss(
        action_stats=action_stats,
        swing_weight=swing_weight,
        device=str(device),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        metrics = _run_epoch(model, loader, loss_fn, optimizer, device, train=True)
        scheduler.step()

        if verbose and (epoch % 50 == 0 or epoch == 1 or epoch <= 5):
            elapsed = time.time() - t0
            print(f"  ep {epoch:4d} | loss={metrics['total']:.4f} "
                  f"action={metrics['action']:.4f} done_acc={metrics['done_acc']:.3f} "
                  f"t={elapsed:.1f}s", flush=True)

        if metrics['action'] < target_loss:
            elapsed = time.time() - t0
            print(f"\n  PASS — action_loss={metrics['action']:.4f} < {target_loss} "
                  f"at epoch {epoch}  ({elapsed:.1f}s)", flush=True)
            return {'status': 'PASS', 'epoch': epoch, 'action_loss': metrics['action'],
                    'elapsed': elapsed}

    elapsed = time.time() - t0
    print(f"\n  FAIL — action_loss={metrics['action']:.4f} after {max_epochs} epochs "
          f"({elapsed:.1f}s)", flush=True)
    return {'status': 'FAIL', 'epoch': max_epochs, 'action_loss': metrics['action'],
            'elapsed': elapsed}


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train_full(
    arch:          str,
    device:        torch.device,
    repo_path:     str,
    action_stats:  dict,
    out_dir:       Path,
    n_epochs:      int   = 20,
    batch_size:    int   = 64,
    lr:            float = 3e-4,
    swing_weight:  float = 2.0,
    train_fraction: float = 0.9,
    num_workers:   int   = 0,
    resume_ckpt:   str | None = None,
) -> list:
    """Runs the full training loop with per-epoch checkpoints.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: Path to the dataset repo (e.g. easy_train80).
        action_stats: Action normalization statistics (mean/std/etc.).
        out_dir: Directory to write checkpoints and logs to.
        n_epochs: Number of training epochs.
        batch_size: Batch size for train/val loaders.
        lr: Learning rate for the AdamW optimizer.
        swing_weight: Swing-joint upweighting factor for GaitFixLoss.
        train_fraction: Fraction of frames used for the train split.
        num_workers: Number of DataLoader worker processes.
        resume_ckpt: Optional checkpoint path to resume training from.

    Returns:
        A list of per-epoch metric dicts (epoch, train, val, elapsed_s).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders (vision=zeros via load_video=False → zeros returned)
    train_loader = make_dataloader(
        mode='parquet', split='train', batch_size=batch_size,
        repo_path=repo_path, num_workers=num_workers,
        train_fraction=train_fraction,
        load_video=False,   # zeros for vision — not the stability bottleneck
    )
    val_loader = make_dataloader(
        mode='parquet', split='val', batch_size=batch_size,
        repo_path=repo_path, num_workers=num_workers,
        train_fraction=train_fraction,
        load_video=False,
    )

    model = GroundedNav(arch=arch, teacher_forcing=True, chunk_H=1).to(device)
    loss_fn = GaitFixLoss(
        action_stats=action_stats,
        swing_weight=swing_weight,
        device=str(device),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    start_epoch = 1
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[train_gaitfix] Resumed from {resume_ckpt}, continuing from epoch {start_epoch}")

    # Pack action stats as serializable dict for checkpoint
    action_stats_serial = {
        'mean':           list(map(float, action_stats['mean'])),
        'std':            list(map(float, action_stats['std'])),
        'default_angles': list(map(float, action_stats['default_angles'])),
        'n_frames':       int(action_stats['n_frames']),
    }

    # Save action stats JSON alongside checkpoints
    with open(out_dir / 'action_stats.json', 'w') as f:
        json.dump(action_stats_serial, f, indent=2)
    print(f"[train_gaitfix] Action stats saved to {out_dir}/action_stats.json")

    log = []
    best_loss = float('inf')

    print(f"\n[train_gaitfix] Starting training — arch={arch}  epochs={n_epochs}  "
          f"batch={batch_size}  lr={lr}  swing_weight={swing_weight}", flush=True)
    print(f"  train frames: {len(train_loader.dataset)}  "
          f"val frames: {len(val_loader.dataset)}", flush=True)

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        tr  = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val = _run_epoch(model, val_loader,   loss_fn, None,      device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        row = {'epoch': epoch, 'train': tr, 'val': val, 'elapsed_s': elapsed}
        log.append(row)

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"tr_loss={tr['total']:.4f} tr_act={tr['action']:.4f} "
            f"tr_vel={tr['vel']:.4f} done_acc={tr['done_acc']:.3f} | "
            f"val_loss={val['total']:.4f} val_act={val['action']:.4f} "
            f"val_vel={val['vel']:.4f} | "
            f"t={elapsed:.1f}s",
            flush=True,
        )

        # Save per-epoch checkpoint (WITH action_stats embedded)
        epoch_ckpt = {
            'epoch':          epoch,
            'arch':           arch,
            'model_state':    model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'action_stats':   action_stats_serial,
            'swing_weight':   swing_weight,
            'train':          tr,
            'val':            val,
        }
        torch.save(epoch_ckpt, out_dir / f'epoch_{epoch:04d}.pt')
        # Rolling latest
        torch.save(epoch_ckpt, out_dir / 'model.pt')

        # Best by val action loss
        ref_loss = val.get('action', val.get('total', tr['action']))
        if ref_loss < best_loss:
            best_loss = ref_loss
            torch.save(epoch_ckpt, out_dir / 'model_best.pt')
            print(f"  -> New best val_action={ref_loss:.4f}, saved model_best.pt", flush=True)

    # Save training curves
    with open(out_dir / 'curves.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n[train_gaitfix] Training complete. Best val_action={best_loss:.4f}", flush=True)
    return log


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parses CLI args and runs the overfit gate, training, and/or audit."""
    ap = argparse.ArgumentParser(description='GaitFix trainer — Fix 1 residual+standardized action')
    ap.add_argument('--arch',      default='A', choices=['A', 'C'])
    ap.add_argument('--data',      required=True, help='Dataset repo path (easy_train80)')
    ap.add_argument('--out',       default='runs/gaitfix_A', help='Output dir')
    ap.add_argument('--epochs',    type=int,   default=20)
    ap.add_argument('--batch',     type=int,   default=64)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--swing-weight', type=float, default=2.0,
                    help='Loss weight multiplier for top-5 highest-variance (swing) joints')
    ap.add_argument('--overfit',   action='store_true',
                    help='Run overfit gate on a small real-data subset before full training')
    ap.add_argument('--overfit-samples', type=int, default=32)
    ap.add_argument('--overfit-only',    action='store_true',
                    help='Run overfit gate ONLY, skip full training')
    ap.add_argument('--device',    default='auto')
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--resume-ckpt', default=None, help='Resume from epoch checkpoint')
    ap.add_argument('--audit-vel', action='store_true',
                    help='Audit velocity head predictions vs GT on val set (Fix 2 diagnostic)')
    ap.add_argument('--audit-ckpt', default=None,
                    help='Checkpoint to load for velocity audit (default: model_best.pt in --out)')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[train_gaitfix] Device: {device}", flush=True)

    out_dir = Path(args.out)

    # --- Step 0: compute action statistics ---
    stats_path = out_dir / 'action_stats.json'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[train_gaitfix] Computing action statistics over training set...", flush=True)
    action_stats = compute_action_stats(
        repo_path=args.data,
        train_fraction=0.9,
        stats_path=str(stats_path),
        verbose=True,
    )

    # --- Step 1: overfit gate (mandatory before full training) ---
    if args.overfit or args.overfit_only:
        gate_result = run_overfit_gate(
            arch=args.arch,
            device=device,
            repo_path=args.data,
            action_stats=action_stats,
            batch_size=min(args.batch, args.overfit_samples),
            overfit_n=args.overfit_samples,
            swing_weight=args.swing_weight,
        )
        print(f"\n[train_gaitfix] Overfit gate: {gate_result['status']}", flush=True)
        if gate_result['status'] == 'FAIL':
            print("[train_gaitfix] WARN: overfit gate FAILED — model may not converge. Continuing anyway.")

        if args.overfit_only:
            print(json.dumps(gate_result, indent=2))
            return

    # --- Step 2: full training ---
    log = train_full(
        arch=args.arch,
        device=device,
        repo_path=args.data,
        action_stats=action_stats,
        out_dir=out_dir,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        swing_weight=args.swing_weight,
        num_workers=args.num_workers,
        resume_ckpt=args.resume_ckpt,
    )

    # --- Step 3: velocity head audit (Fix 2 diagnostic) ---
    if args.audit_vel:
        ckpt_path = args.audit_ckpt or str(out_dir / 'model_best.pt')
        print(f"\n[train_gaitfix] Fix-2 velocity head audit on {ckpt_path}...", flush=True)
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model = GroundedNav(arch=args.arch, teacher_forcing=True, chunk_H=1).to(device)
            model.load_state_dict(ckpt['model_state'])

            val_loader = make_dataloader(
                mode='parquet', split='val', batch_size=32,
                repo_path=args.data, num_workers=0, load_video=False,
            )
            vel_stats = audit_velocity_head(model, val_loader, device)
            print("\n[Fix-2] Velocity head audit:")
            print(f"  pred vx: mean={vel_stats['pred_mean_vx']:.4f} std={vel_stats['pred_std_vx']:.4f}")
            print(f"  GT   vx: mean={vel_stats['gt_mean_vx']:.4f}   std={vel_stats['gt_std_vx']:.4f}")
            print(f"  pred wz: mean={vel_stats['pred_mean_wz']:.4f} std={vel_stats['pred_std_wz']:.4f}")
            print(f"  GT   wz: mean={vel_stats['gt_mean_wz']:.4f}   std={vel_stats['gt_std_wz']:.4f}")
            print(f"  MAE vx:  {vel_stats['mae_vx']:.4f}")
            print(f"  vel_head_near_zero: {vel_stats['vel_head_near_zero']}")
            with open(out_dir / 'vel_audit.json', 'w') as f:
                json.dump(vel_stats, f, indent=2)
        else:
            print(f"[WARN] Checkpoint not found for audit: {ckpt_path}")


if __name__ == '__main__':
    main()
