"""
action_stats.py — Compute per-joint delta statistics over the training set.

Fix 1 (gait-fix): The student predicts standardized residuals from the default pose:
    delta_j = action_j - default_angles_j
    normalized_delta_j = (delta_j - mean_j) / std_j

This module scans the training parquet files, computes per-joint mean and std of
delta over ALL frames in the training split, and returns/saves them.

The stats are saved alongside the checkpoint (in checkpoint meta) so deploy-time
de-normalization is exact.

Usage:
    from code.action_stats import compute_action_stats, load_action_stats
    stats = compute_action_stats('dataset/easy_train80', train_fraction=0.9)
    # stats = {'mean': (15,), 'std': (15,), 'default_angles': (15,)}

RF-1: moved from code/action_stats.py to code/policy/action_stats.py (see
code/action_stats.py, the old-path compat alias, and docs/refactor_plan.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Default joint angles from teacher.py (must match exactly)
DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0,  0.3, -0.2, 0.0,   # left  leg (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
    -0.1, 0.0, 0.0,  0.3, -0.2, 0.0,   # right leg
     0.0, 0.0, 0.0,                      # waist (yaw, roll, pitch)
], dtype=np.float32)

# Minimum std floor to avoid division by zero on near-constant joints (e.g. waist_yaw)
STD_FLOOR = 1e-3


def compute_action_stats(
    repo_path: str,
    train_fraction: float = 0.9,
    stats_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Scan the parquet training split and compute per-joint delta statistics.

    Args:
        repo_path: Path to dataset directory (easy_train80, etc.).
        train_fraction: Fraction of episodes used for training (same as
            training split).
        stats_path: If provided, save JSON to this path.
        verbose: Print progress.

    Returns:
        dict with keys:
            'mean'           : np.float32 (15,)  per-joint mean of delta
            'std'            : np.float32 (15,)  per-joint std  of delta (floored at STD_FLOOR)
            'default_angles' : np.float32 (15,)  default pose used
            'n_frames'       : int               total frames scanned
            'repo_path'      : str

    Raises:
        FileNotFoundError: If no episodes are found in `repo_path`.
        RuntimeError: If no valid parquet files are found in the training split.
    """
    import pandas as pd

    repo = Path(repo_path)
    meta_path = repo / 'meta' / 'episodes.jsonl'

    # Discover episodes
    episodes_meta = []
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes_meta.append(json.loads(line))
    else:
        chunk_dir = repo / 'data' / 'chunk-000'
        if chunk_dir.exists():
            for p in sorted(chunk_dir.glob('episode_*.parquet')):
                ep_id = int(p.stem.split('_')[1])
                episodes_meta.append({'episode_index': ep_id})

    if not episodes_meta:
        raise FileNotFoundError(f"No episodes found in {repo_path}")

    # Train split (first train_fraction episodes)
    n_ep    = len(episodes_meta)
    n_train = max(1, int(n_ep * train_fraction))
    train_eps = episodes_meta[:n_train]

    if verbose:
        print(f"[action_stats] Scanning {n_train}/{n_ep} train episodes in {repo_path}")

    all_deltas = []  # list of (N, 15) arrays

    for em in train_eps:
        ep_id     = em.get('episode_index', em.get('episode_id', 0))
        chunk_idx = ep_id // 1000
        parq      = repo / 'data' / f'chunk-{chunk_idx:03d}' / f'episode_{ep_id:06d}.parquet'
        if not parq.exists():
            if verbose:
                print(f"  [WARN] Missing parquet for ep {ep_id}, skipping")
            continue

        df = pd.read_parquet(parq)
        # action column is stored as list/array per row — shape (N, 15)
        actions = np.stack([np.array(r, dtype=np.float32) for r in df['action']])  # (N, 15)
        delta   = actions - DEFAULT_ANGLES[np.newaxis, :]                           # (N, 15)
        all_deltas.append(delta)

    if not all_deltas:
        raise RuntimeError("No valid parquet files found in training split")

    all_delta = np.concatenate(all_deltas, axis=0)  # (total_frames, 15)
    n_frames  = all_delta.shape[0]

    mean  = all_delta.mean(axis=0).astype(np.float32)   # (15,)
    std   = all_delta.std(axis=0).astype(np.float32)    # (15,)
    std   = np.maximum(std, STD_FLOOR)                  # floor to avoid /0

    if verbose:
        print(f"[action_stats] Total frames: {n_frames}")
        print(f"[action_stats] Delta mean (rad): {np.round(mean, 4)}")
        print(f"[action_stats] Delta std  (rad): {np.round(std,  4)}")
        # Identify swing joints (higher variance joints — likely hip_pitch, knee)
        sorted_idx = np.argsort(std)[::-1]
        joint_names = [
            'l_hip_pitch','l_hip_roll','l_hip_yaw','l_knee','l_ankle_pitch','l_ankle_roll',
            'r_hip_pitch','r_hip_roll','r_hip_yaw','r_knee','r_ankle_pitch','r_ankle_roll',
            'waist_yaw','waist_roll','waist_pitch',
        ]
        print(f"[action_stats] Top-5 highest-variance joints (swing candidates):")
        for idx in sorted_idx[:5]:
            print(f"  [{idx:2d}] {joint_names[idx]:<20} std={std[idx]:.4f} rad")

    stats = {
        'mean':           mean.tolist(),
        'std':            std.tolist(),
        'default_angles': DEFAULT_ANGLES.tolist(),
        'n_frames':       int(n_frames),
        'repo_path':      str(repo_path),
    }

    if stats_path is not None:
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        if verbose:
            print(f"[action_stats] Saved to {stats_path}")

    return stats


def load_action_stats(stats_path: str) -> dict:
    """Load stats JSON and return arrays as np.float32.

    Args:
        stats_path: Path to a JSON file previously saved by `compute_action_stats`.

    Returns:
        dict with 'mean', 'std', 'default_angles' (np.float32 arrays), plus
        'n_frames' (int) and 'repo_path' (str).
    """
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        'mean':           np.array(raw['mean'],           dtype=np.float32),
        'std':            np.array(raw['std'],            dtype=np.float32),
        'default_angles': np.array(raw['default_angles'], dtype=np.float32),
        'n_frames':       raw['n_frames'],
        'repo_path':      raw.get('repo_path', ''),
    }


def stats_from_checkpoint(ckpt: dict) -> dict:
    """Extract action stats from a checkpoint dict (saved by train_gaitfix.py).

    Args:
        ckpt: Checkpoint dict as produced by train_gaitfix.py.

    Returns:
        dict with 'mean', 'std', 'default_angles' (np.float32 arrays) and
        'n_frames' (int).

    Raises:
        KeyError: If `ckpt` has no 'action_stats' entry (or it lacks 'mean'),
            i.e. it was not trained with train_gaitfix.py.
    """
    meta = ckpt.get('action_stats', {})
    if not meta or 'mean' not in meta:
        raise KeyError(
            "Checkpoint does not contain 'action_stats' — was it trained with train_gaitfix.py?")
    return {
        'mean':           np.array(meta['mean'],           dtype=np.float32),
        'std':            np.array(meta['std'],            dtype=np.float32),
        'default_angles': np.array(meta['default_angles'], dtype=np.float32),
        'n_frames':       meta.get('n_frames', 0),
    }
