"""code.train.gaitfix — Gait-fix training run: Fix 1 (residual + standardized action target).

CLI entry point + overfit gate + full-training driver. See
``code.train.gaitfix_loss`` for the ``GaitFixLoss`` definition and
``code.train.gaitfix_epoch`` for the shared per-epoch runner and the Fix-2
velocity-head audit (both re-exported here so ``from code.train_gaitfix import
GaitFixLoss, _run_epoch`` — the pre-RF-1 import path — keeps working via the
old-path alias).

Architecture: Arch A, vision=ZEROS (zeros fed as RGB — not the stability
issue; keeps training fast). Runs on the 80-episode easy dataset.

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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from code.action_stats import compute_action_stats
from code.dataset import make_dataloader, ParquetDataset
from code.small_vla import GroundedNav

# Re-exported for the pre-RF-1 import path (`from code.train_gaitfix import
# GaitFixLoss, _run_epoch, audit_velocity_head`) via the old-path alias shim.
from code.train.gaitfix_loss import GaitFixLoss, JOINT_NAMES
from code.train.gaitfix_epoch import _run_epoch, audit_velocity_head

__all__ = [
    'JOINT_NAMES', 'GaitFixLoss', '_run_epoch', 'audit_velocity_head',
    'run_overfit_gate', 'train_full', 'main',
]


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
