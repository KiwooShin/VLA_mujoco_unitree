"""
code.train.dart_phase — Training with DART data + gait-phase input (Fix 4+5).

Builds on ``code.train.gaitfix`` (Fix 1 residual/standardized actions) and adds:
  Fix 4: gait-phase [sin(phi), cos(phi)] appended to proprio → proprio_dim 55 → 57
  Fix 5: DART-augmented training data (noisy execution + clean labels)

Model: GroundedNav Arch A with proprio_dim=57.
Loss:  GaitFixLoss (residual standardized action targets, swing-joint upweighting).
Data:  Combined DART + clean dataset (both with phase column) from dataset_phase.py.

Usage
-----
# Overfit gate (MANDATORY first check):
MUJOCO_GL=egl python code/train_dart_phase.py \\
    --data dataset/dart_combined \\
    --out runs/dart_phase_A \\
    --overfit --overfit-only

# Full training (20 epochs, per-epoch checkpoints):
MUJOCO_GL=egl python code/train_dart_phase.py \\
    --data dataset/dart_combined \\
    --out runs/dart_phase_A \\
    --epochs 20 --batch 64 --lr 3e-4
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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from code.action_stats import compute_action_stats
from code.dataset_phase import make_phase_dataloader, PhaseParquetDataset, PROPRIO_DIM_PHASE
from code.small_vla import GroundedNav
from code.train.gaitfix_loss import GaitFixLoss
from code.train.gaitfix_epoch import _run_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROPRIO_DIM: int = PROPRIO_DIM_PHASE   # 57


# ---------------------------------------------------------------------------
# Overfit gate (small subset of real DART+phase data)
# ---------------------------------------------------------------------------
def run_overfit_gate(
    arch:         str,
    device:       torch.device,
    repo_path:    str,
    action_stats: dict,
    batch_size:   int   = 16,
    overfit_n:    int   = 32,
    max_epochs:   int   = 300,
    target_loss:  float = 0.10,
    lr:           float = 3e-4,
    swing_weight: float = 2.0,
    verbose:      bool  = True,
) -> dict:
    """Runs the DART+phase overfit gate on a small subset of real data.

    Trains ``GroundedNav`` (arch ``A`` or ``C``, proprio_dim=57) on a small
    fixed subset until the action loss drops below ``target_loss`` or
    ``max_epochs`` is reached.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: Path to the combined DART+clean dataset repo.
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
    print(f"  DART+PHASE OVERFIT GATE — Arch {arch}  proprio_dim={PROPRIO_DIM}")
    print(f"  repo={repo_path}  n_samples={overfit_n}  target_loss={target_loss}")
    print(f"{'='*60}")

    full_ds = PhaseParquetDataset(
        repo_paths=[repo_path], split='train', train_fraction=0.9,
    )
    n = min(overfit_n, len(full_ds))
    subset = Subset(full_ds, list(range(n)))
    loader = DataLoader(subset, batch_size=min(batch_size, n), shuffle=False, num_workers=0)
    print(f"  Overfit subset: {n} samples from {len(full_ds)} total train frames")

    model = GroundedNav(
        arch=arch, teacher_forcing=True, chunk_H=1,
        proprio_dim=PROPRIO_DIM,   # 57
    ).to(device)
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

        if verbose and (epoch % 50 == 0 or epoch <= 5):
            elapsed = time.time() - t0
            print(f"  ep {epoch:4d} | action={metrics['action']:.4f} "
                  f"total={metrics['total']:.4f} t={elapsed:.1f}s", flush=True)

        if metrics['action'] < target_loss:
            elapsed = time.time() - t0
            print(f"\n  PASS — action_loss={metrics['action']:.4f} < {target_loss} "
                  f"at epoch {epoch}  ({elapsed:.1f}s)", flush=True)
            return {'status': 'PASS', 'epoch': epoch,
                    'action_loss': metrics['action'], 'elapsed': elapsed}

    elapsed = time.time() - t0
    print(f"\n  FAIL — action_loss={metrics['action']:.4f} after {max_epochs} epochs", flush=True)
    return {'status': 'FAIL', 'epoch': max_epochs,
            'action_loss': metrics['action'], 'elapsed': elapsed}


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------
def train_full(
    arch:           str,
    device:         torch.device,
    repo_path:      str,
    action_stats:   dict,
    out_dir:        Path,
    n_epochs:       int   = 20,
    batch_size:     int   = 64,
    lr:             float = 3e-4,
    swing_weight:   float = 2.0,
    train_fraction: float = 0.9,
    num_workers:    int   = 0,
    resume_ckpt:    str | None = None,
    reset_epoch:    bool  = False,
) -> list:
    """Runs full training over the combined DART+phase dataset.

    Trains ``GroundedNav`` with ``GaitFixLoss``, saving a per-epoch
    checkpoint plus a running-best checkpoint to ``out_dir``, and a
    ``curves.json`` log of per-epoch train/val metrics.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: Path to the combined DART+phase dataset repo.
        action_stats: Action normalization statistics (mean/std/etc.).
        out_dir: Directory to write checkpoints and logs to.
        n_epochs: Number of training epochs.
        batch_size: Batch size for train/val loaders.
        lr: Learning rate for the AdamW optimizer.
        swing_weight: Swing-joint upweighting factor for GaitFixLoss.
        train_fraction: Fraction of frames used for the train split.
        num_workers: Number of DataLoader worker processes.
        resume_ckpt: Optional checkpoint path to resume training from.
        reset_epoch: If True, reset the epoch counter to 1 when resuming
            (used for fine-tuning from a pre-trained checkpoint).

    Returns:
        A list of per-epoch metric dicts (epoch, train, val, elapsed_s).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_phase_dataloader(
        repo_paths=[repo_path], split='train',
        batch_size=batch_size, train_fraction=train_fraction,
        num_workers=num_workers,
    )
    val_loader = make_phase_dataloader(
        repo_paths=[repo_path], split='val',
        batch_size=batch_size, train_fraction=train_fraction,
        num_workers=num_workers,
    )

    model = GroundedNav(
        arch=arch, teacher_forcing=True, chunk_H=1,
        proprio_dim=PROPRIO_DIM,   # 57
    ).to(device)
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
        model.load_state_dict(ckpt['model_state'], strict=False)
        if not reset_epoch and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if reset_epoch:
            start_epoch = 1
            print(f"[train_dart_phase] Fine-tuning from {resume_ckpt} (epoch counter reset to 1)")
        else:
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"[train_dart_phase] Resumed from {resume_ckpt}, epoch {start_epoch}")

    action_stats_serial = {
        'mean':           list(map(float, action_stats['mean'])),
        'std':            list(map(float, action_stats['std'])),
        'default_angles': list(map(float, action_stats['default_angles'])),
        'n_frames':       int(action_stats['n_frames']),
    }
    with open(out_dir / 'action_stats.json', 'w') as f:
        json.dump(action_stats_serial, f, indent=2)

    log = []
    best_loss = float('inf')

    print(f"\n[train_dart_phase] Starting training — arch={arch}  epochs={n_epochs}  "
          f"proprio_dim={PROPRIO_DIM}  batch={batch_size}  lr={lr}", flush=True)
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
            f"tr_act={tr['action']:.4f} tr_vel={tr['vel']:.4f} "
            f"done_acc={tr['done_acc']:.3f} | "
            f"val_act={val['action']:.4f} val_vel={val['vel']:.4f} | "
            f"t={elapsed:.1f}s",
            flush=True,
        )

        # Save checkpoint
        epoch_ckpt = {
            'epoch':          epoch,
            'arch':           arch,
            'proprio_dim':    PROPRIO_DIM,
            'model_state':    model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'action_stats':   action_stats_serial,
            'swing_weight':   swing_weight,
            'train':          tr,
            'val':            val,
            'dart_phase':     True,   # marker so inferencer knows to use 57-d proprio
        }
        torch.save(epoch_ckpt, out_dir / f'epoch_{epoch:04d}.pt')
        torch.save(epoch_ckpt, out_dir / 'model.pt')

        ref_loss = val.get('action', val.get('total', tr['action']))
        if ref_loss < best_loss:
            best_loss = ref_loss
            torch.save(epoch_ckpt, out_dir / 'model_best.pt')
            print(f"  -> New best val_action={ref_loss:.4f}, saved model_best.pt", flush=True)

    with open(out_dir / 'curves.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n[train_dart_phase] Done. Best val_action={best_loss:.4f}", flush=True)
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Parses CLI args and runs the DART+phase overfit gate and/or training."""
    ap = argparse.ArgumentParser(
        description='DART+Phase trainer (Fix 4+5 on top of Fix 1 residual actions)'
    )
    ap.add_argument('--arch',         default='A', choices=['A', 'C'])
    ap.add_argument('--data',         required=True,
                    help='Combined DART+clean dataset dir (dart_combined)')
    ap.add_argument('--out',          default='runs/dart_phase_A')
    ap.add_argument('--epochs',       type=int,   default=20)
    ap.add_argument('--batch',        type=int,   default=64)
    ap.add_argument('--lr',           type=float, default=3e-4)
    ap.add_argument('--swing-weight', type=float, default=2.0)
    ap.add_argument('--overfit',      action='store_true')
    ap.add_argument('--overfit-samples', type=int, default=32)
    ap.add_argument('--overfit-only', action='store_true')
    ap.add_argument('--device',       default='auto')
    ap.add_argument('--num-workers',  type=int, default=0)
    ap.add_argument('--resume-ckpt',  default=None)
    ap.add_argument('--reset-epoch',  action='store_true',
                    help='Reset epoch counter to 1 when resuming (for fine-tuning from pre-trained ckpt)')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[train_dart_phase] Device: {device}  proprio_dim={PROPRIO_DIM}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute action stats from the combined dataset (action column same format)
    print(f"\n[train_dart_phase] Computing action stats over {args.data}...", flush=True)
    stats_path = out_dir / 'action_stats.json'
    action_stats = compute_action_stats(
        repo_path=args.data,
        train_fraction=0.9,
        stats_path=str(stats_path),
        verbose=True,
    )

    if args.overfit or args.overfit_only:
        gate = run_overfit_gate(
            arch=args.arch,
            device=device,
            repo_path=args.data,
            action_stats=action_stats,
            batch_size=min(args.batch, args.overfit_samples),
            overfit_n=args.overfit_samples,
            swing_weight=args.swing_weight,
        )
        print(f"\n[train_dart_phase] Overfit gate: {gate['status']}", flush=True)
        if args.overfit_only:
            print(json.dumps(gate, indent=2))
            return

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
        reset_epoch=getattr(args, 'reset_epoch', False),
    )


if __name__ == '__main__':
    main()
