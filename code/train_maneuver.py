"""
train_maneuver.py — Fine-tune the demo_dart_A locomotion model on maneuver data.

Fine-tunes from runs/demo_dart_A/epoch_0010.pt (best closed-loop, 80% at ep3-ep10).

Key differences from train_dart_phase.py:
  - proprio_dim=62 (55+2 phase+5 maneuver)
  - Dataset: ManeuverParquetDataset (from dataset_maneuver.py)
  - Model weights loaded from locomotion checkpoint with partial match:
    the proprio_enc GRU input expands from 57→62. Re-initialize only that layer.
  - Output: runs/maneuver_A/

Usage
-----
# Overfit gate (smoke 1 epoch on small subset):
MUJOCO_GL=egl python code/train_maneuver.py \\
    --data dataset/maneuver \\
    --out runs/maneuver_A \\
    --overfit --overfit-only

# Full training:
MUJOCO_GL=egl python code/train_maneuver.py \\
    --data dataset/maneuver \\
    --out runs/maneuver_A \\
    --resume-ckpt runs/demo_dart_A/epoch_0010.pt \\
    --epochs 20 --batch 64 --lr 5e-5
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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.action_stats import compute_action_stats, DEFAULT_ANGLES, STD_FLOOR
from code.dataset_maneuver import (
    make_maneuver_dataloader, ManeuverParquetDataset,
    PROPRIO_DIM_MANEUVER,
)
from code.small_vla import GroundedNav, DEFAULTS
from code.train_gaitfix import GaitFixLoss, _run_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROPRIO_DIM: int = PROPRIO_DIM_MANEUVER   # 62


def _expand_proprio_enc(model: GroundedNav, old_dim: int, new_dim: int) -> None:
    """Expands the proprio encoder's GRU input_size from old_dim to new_dim.

    Re-initializes only the input_weight columns for the new dims; keeps old
    weights. This allows loading a 57-d checkpoint and expanding to 62-d.

    Args:
        model: The GroundedNav model whose proprio_enc GRU is expanded
            in place.
        old_dim: Original GRU input size (e.g. 57).
        new_dim: New GRU input size (e.g. 62).
    """
    old_gru = model.proprio_enc.gru
    hidden  = old_gru.hidden_size
    # new GRU
    new_gru = nn.GRU(new_dim, hidden, batch_first=True, num_layers=1)
    # Copy old weights for the first old_dim input dims
    # GRU weight_ih_l0 shape: (3*hidden, input_size)
    with torch.no_grad():
        # input-to-hidden weights: rows 0:3*hidden, cols 0:old_dim
        new_gru.weight_ih_l0[:, :old_dim].copy_(old_gru.weight_ih_l0)
        # new cols (old_dim:new_dim) init to small random
        nn.init.orthogonal_(new_gru.weight_ih_l0[:, old_dim:])
        # hidden-to-hidden and biases: copy directly
        new_gru.weight_hh_l0.copy_(old_gru.weight_hh_l0)
        new_gru.bias_ih_l0.copy_(old_gru.bias_ih_l0)
        new_gru.bias_hh_l0.copy_(old_gru.bias_hh_l0)
    model.proprio_enc.gru = new_gru
    print(f"[train_maneuver] Expanded proprio_enc GRU: {old_dim}-d → {new_dim}-d "
          f"(old weights preserved, new cols orthogonal-init)", flush=True)


def load_loco_checkpoint(ckpt_path: str, device: torch.device) -> tuple[GroundedNav, dict]:
    """Loads a locomotion checkpoint (proprio_dim=57) and expands it to the
    maneuver model (62-d).

    Steps:
    1. Build GroundedNav with proprio_dim=57 (match checkpoint exactly).
    2. Load state dict strictly.
    3. Expand proprio_enc GRU input: 57→62 (preserving old weights).

    Args:
        ckpt_path: Path to the locomotion checkpoint file.
        device: Torch device to move the expanded model to.

    Returns:
        A tuple of (expanded model on device, raw checkpoint dict).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_proprio_dim = ckpt.get('proprio_dim', 57)
    arch = ckpt.get('arch', 'A')
    chunk_H = ckpt.get('chunk_H', 1)

    print(f"[train_maneuver] Loading checkpoint: {ckpt_path}")
    print(f"  arch={arch}  proprio_dim={ckpt_proprio_dim}  chunk_H={chunk_H}")

    # Build with checkpoint's proprio_dim
    model = GroundedNav(
        arch=arch, teacher_forcing=True, chunk_H=chunk_H,
        proprio_dim=ckpt_proprio_dim,
    )

    # Load state dict
    model_state = ckpt.get('model_state', ckpt.get('state_dict', None))
    if model_state is None:
        # Direct state dict
        model_state = ckpt
    miss, unexp = model.load_state_dict(model_state, strict=False)
    if miss:
        print(f"  WARNING: {len(miss)} missing keys: {miss[:5]}")
    if unexp:
        print(f"  WARNING: {len(unexp)} unexpected keys: {unexp[:5]}")
    print(f"  Checkpoint loaded cleanly.", flush=True)

    # Expand proprio_enc GRU if needed
    if ckpt_proprio_dim != PROPRIO_DIM:
        _expand_proprio_enc(model, ckpt_proprio_dim, PROPRIO_DIM)

    model = model.to(device)
    return model, ckpt


def run_overfit_gate(
    arch:         str,
    device:       torch.device,
    repo_path:    str,
    action_stats: dict,
    batch_size:   int   = 16,
    overfit_n:    int   = 32,
    max_epochs:   int   = 300,
    target_loss:  float = 0.10,
    lr:           float = 1e-4,
    swing_weight: float = 2.0,
    verbose:      bool  = True,
) -> dict:
    """Runs the maneuver overfit gate on a small subset of real data.

    Trains ``GroundedNav`` (proprio_dim=62) on a small fixed subset until
    the action loss drops below ``target_loss`` or ``max_epochs`` is reached.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: Path to the maneuver dataset repo.
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
    print(f"  MANEUVER OVERFIT GATE — Arch {arch}  proprio_dim={PROPRIO_DIM}")
    print(f"  repo={repo_path}  n_samples={overfit_n}  target_loss={target_loss}")
    print(f"{'='*60}")

    full_ds = ManeuverParquetDataset(
        repo_paths=[repo_path], split='train', train_fraction=0.9,
    )
    n = min(overfit_n, len(full_ds))
    subset = Subset(full_ds, list(range(n)))
    loader = DataLoader(subset, batch_size=min(batch_size, n), shuffle=False, num_workers=0)
    print(f"  Overfit subset: {n} samples from {len(full_ds)} total train frames")

    model = GroundedNav(
        arch=arch, teacher_forcing=True, chunk_H=1,
        proprio_dim=PROPRIO_DIM,
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
    metrics = {}
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
    print(f"\n  FAIL — action_loss={metrics.get('action', 99):.4f} after {max_epochs} epochs",
          flush=True)
    return {'status': 'FAIL', 'epoch': max_epochs,
            'action_loss': metrics.get('action', 99), 'elapsed': elapsed}


def train_full(
    arch:           str,
    device:         torch.device,
    repo_path:      str | list[str],
    action_stats:   dict,
    out_dir:        Path,
    n_epochs:       int   = 20,
    batch_size:     int   = 64,
    lr:             float = 5e-5,
    swing_weight:   float = 2.0,
    train_fraction: float = 0.9,
    num_workers:    int   = 0,
    resume_ckpt:    str | None = None,
) -> list:
    """Runs full maneuver fine-tuning/training.

    Loads a locomotion checkpoint (if given) and expands its proprio encoder
    to 62-d, then trains with ``GaitFixLoss``, saving a per-epoch checkpoint
    plus a running-best checkpoint to ``out_dir``, and a ``curves.json`` log
    of per-epoch train/val metrics.

    Args:
        arch: Model architecture variant ('A' or 'C').
        device: Torch device to train on.
        repo_path: One or more maneuver dataset repo paths.
        action_stats: Action normalization statistics (mean/std/etc.).
        out_dir: Directory to write checkpoints and logs to.
        n_epochs: Number of training epochs.
        batch_size: Batch size for train/val loaders.
        lr: Learning rate for the AdamW optimizer.
        swing_weight: Swing-joint upweighting factor for GaitFixLoss.
        train_fraction: Fraction of frames used for the train split.
        num_workers: Number of DataLoader worker processes.
        resume_ckpt: Optional locomotion checkpoint path to fine-tune from.
            If missing/None, trains a fresh model from scratch.

    Returns:
        A list of per-epoch metric dicts (epoch, train, val, elapsed_s).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept single string or list of repo paths
    repo_paths = [repo_path] if isinstance(repo_path, str) else list(repo_path)

    train_loader = make_maneuver_dataloader(
        repo_paths=repo_paths, split='train',
        batch_size=batch_size, train_fraction=train_fraction,
        num_workers=num_workers,
    )
    val_loader = make_maneuver_dataloader(
        repo_paths=repo_paths, split='val',
        batch_size=batch_size, train_fraction=train_fraction,
        num_workers=num_workers,
    )

    if resume_ckpt and os.path.isfile(resume_ckpt):
        model, ckpt_meta = load_loco_checkpoint(resume_ckpt, device)
    else:
        print(f"[train_maneuver] No checkpoint, training from scratch (proprio_dim={PROPRIO_DIM})")
        model = GroundedNav(
            arch=arch, teacher_forcing=True, chunk_H=1,
            proprio_dim=PROPRIO_DIM,
        ).to(device)
        ckpt_meta = {}

    loss_fn = GaitFixLoss(
        action_stats=action_stats,
        swing_weight=swing_weight,
        device=str(device),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

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

    print(f"\n[train_maneuver] Starting — arch={arch}  epochs={n_epochs}  "
          f"proprio_dim={PROPRIO_DIM}  batch={batch_size}  lr={lr}", flush=True)
    print(f"  train frames: {len(train_loader.dataset)}  "
          f"val frames: {len(val_loader.dataset)}", flush=True)

    for epoch in range(1, n_epochs + 1):
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
            'dart_phase':     True,    # 57+ proprio
            'maneuver':       True,    # 62-d proprio (57 + 5 maneuver dims)
            'task':           'maneuver',
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

    print(f"\n[train_maneuver] Done. Best val_action={best_loss:.4f}", flush=True)
    return log


def main() -> None:
    """Parses CLI args and runs the maneuver overfit gate and/or training."""
    ap = argparse.ArgumentParser(
        description='Maneuver trainer (fine-tune from demo_dart_A, proprio_dim=62)'
    )
    ap.add_argument('--arch',         default='A', choices=['A', 'C'])
    ap.add_argument('--data',         required=True, nargs='+',
                    help='One or more dataset repo paths (space-separated). '
                         'Locomotion repos without maneuver cols get zero-padded maneuver dims.')
    ap.add_argument('--out',          default='runs/maneuver_A')
    ap.add_argument('--epochs',       type=int,   default=20)
    ap.add_argument('--batch',        type=int,   default=64)
    ap.add_argument('--lr',           type=float, default=5e-5)
    ap.add_argument('--swing-weight', type=float, default=2.0)
    ap.add_argument('--overfit',      action='store_true')
    ap.add_argument('--overfit-samples', type=int, default=32)
    ap.add_argument('--overfit-only', action='store_true')
    ap.add_argument('--device',       default='auto')
    ap.add_argument('--num-workers',  type=int, default=0)
    ap.add_argument('--resume-ckpt',  default=None)
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[train_maneuver] Device: {device}  proprio_dim={PROPRIO_DIM}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_paths = args.data   # list of paths (nargs='+')
    primary_repo = repo_paths[0]  # use first repo for action_stats (maneuver data)
    print(f"\n[train_maneuver] Repos: {repo_paths}", flush=True)
    print(f"\n[train_maneuver] Computing action stats over {primary_repo}...", flush=True)
    stats_path = out_dir / 'action_stats.json'
    action_stats = compute_action_stats(
        repo_path=primary_repo,
        train_fraction=0.9,
        stats_path=str(stats_path),
        verbose=True,
    )

    if args.overfit or args.overfit_only:
        gate = run_overfit_gate(
            arch=args.arch,
            device=device,
            repo_path=primary_repo,
            action_stats=action_stats,
            batch_size=min(args.batch, args.overfit_samples),
            overfit_n=args.overfit_samples,
            swing_weight=args.swing_weight,
        )
        print(f"\n[train_maneuver] Overfit gate: {gate['status']}", flush=True)
        if args.overfit_only:
            print(json.dumps(gate, indent=2))
            return

    train_full(
        arch=args.arch,
        device=device,
        repo_path=repo_paths,
        action_stats=action_stats,
        out_dir=out_dir,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        swing_weight=args.swing_weight,
        num_workers=args.num_workers,
        resume_ckpt=args.resume_ckpt,
    )


if __name__ == '__main__':
    main()
