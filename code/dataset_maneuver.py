"""
dataset_maneuver.py — Dataset loader for the maneuver skill.

Extends PhaseParquetDataset to include maneuver-specific conditioning:
  - subgoal_index (0=STRAIGHT, 1=TURN_PHASE, 2=STRAIGHT2) as float
  - cos_target, sin_target (target heading direction)
  - heading_err (current heading error, signed rad)
  - landmark_passed (0/1)

These 5 extra dims are appended to proprio after the gait phase [sin, cos]:
  proprio_dim: 55 (base) + 2 (phase) + 5 (maneuver) = 62

The model is instantiated with proprio_dim=62 when using this dataset.

The maneuver dataset can also be combined with the demo_dart_A locomotion data
(55+2=57 proprio) — in that case, the maneuver dims are zero-padded for locomotion
episodes.

Schema note: all parquet rows for maneuver episodes have:
  - 'phase': [sin_phi, cos_phi]
  - 'subgoal_index': int
  - 'cos_target': float
  - 'sin_target': float
  - 'heading_err': float
  - 'landmark_passed': int

For locomotion episodes (no maneuver columns), zeros are appended.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROPRIO_DIM_BASE     = 55    # standard proprio (no phase, no maneuver)
PHASE_DIM            = 2     # [sin(phi), cos(phi)]
MANEUVER_DIM         = 5     # [subgoal_norm, cos_target, sin_target, heading_err_norm, landmark_passed]
PROPRIO_DIM_PHASE    = PROPRIO_DIM_BASE + PHASE_DIM          # 57 (locomotion model)
PROPRIO_DIM_MANEUVER = PROPRIO_DIM_BASE + PHASE_DIM + MANEUVER_DIM   # 62


def _build_maneuver_features(row) -> np.ndarray:
    """
    Extract 5-d maneuver features from a dataframe row.
    Returns np.float32[5].
    """
    subgoal_idx    = float(row.get("subgoal_index",  0)) / 2.0   # normalize to [0, 1]
    cos_target     = float(row.get("cos_target",     1.0))
    sin_target     = float(row.get("sin_target",     0.0))
    heading_err    = float(row.get("heading_err",    0.0)) / np.pi  # normalize to [-1, 1]
    lm_passed      = float(row.get("landmark_passed", 0))
    return np.array([subgoal_idx, cos_target, sin_target, heading_err, lm_passed],
                    dtype=np.float32)


class ManeuverParquetDataset(Dataset):
    """
    Reads maneuver + optionally locomotion parquet datasets.

    Proprio = 55-d base + 2-d phase + 5-d maneuver = 62-d total.

    For episodes without maneuver columns (locomotion data), the 5-d maneuver
    features are zero-padded.
    """

    def __init__(
        self,
        repo_paths: list[str] | str,
        split: str = 'train',
        train_fraction: float = 0.9,
        proprio_K: int = 6,
        chunk_H: int = 1,
        img_size: int = 128,
        in_ch: int = 3,
        lang_cache_path: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(repo_paths, str):
            repo_paths = [repo_paths]

        self.K = proprio_K
        self.H = chunk_H
        self.in_ch = in_ch
        self.img_size = img_size
        self.proprio_dim = PROPRIO_DIM_MANEUVER   # 62

        import pandas as pd

        self._lang_cache: Optional[dict] = None
        if lang_cache_path and os.path.exists(lang_cache_path):
            import pickle
            with open(lang_cache_path, "rb") as f:
                self._lang_cache = pickle.load(f)
            print(f"[ManeuverDataset] Lang cache: {len(self._lang_cache)} entries")

        all_episodes: list[dict] = []

        for repo_path in repo_paths:
            repo = Path(repo_path)
            meta_path = repo / "meta" / "episodes.jsonl"
            episodes_meta = []

            if meta_path.exists():
                with open(meta_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes_meta.append(json.loads(line))
            else:
                chunk_dir = repo / "data" / "chunk-000"
                if chunk_dir.exists():
                    for p in sorted(chunk_dir.glob("episode_*.parquet")):
                        ep_id = int(p.stem.split("_")[1])
                        episodes_meta.append({"episode_index": ep_id})

            n_ep = len(episodes_meta)
            n_train = max(1, int(n_ep * train_fraction))
            if split == 'train':
                ep_slice = episodes_meta[:n_train]
            else:
                ep_slice = episodes_meta[n_train:]

            for em in ep_slice:
                ep_id   = em.get("episode_index", em.get("episode_id", 0))
                chunk_i = ep_id // 1000
                parq = repo / "data" / f"chunk-{chunk_i:03d}" / f"episode_{ep_id:06d}.parquet"
                if not parq.exists():
                    continue
                df = pd.read_parquet(parq)
                has_phase    = "phase" in df.columns
                has_maneuver = "subgoal_index" in df.columns

                # Pre-convert list columns to numpy arrays for fast __getitem__
                proprio_np = np.stack([np.array(v, dtype=np.float32)
                                       for v in df["proprio"].values])  # (L, 55)
                action_np  = np.stack([np.array(v, dtype=np.float32)
                                       for v in df["action"].values])   # (L, 15)
                goal_np    = np.stack([np.array(v, dtype=np.float32)
                                       for v in df["goal"].values])     # (L, 3)
                vel_np     = np.stack([np.array(v, dtype=np.float32)
                                       for v in df["vel_cmd"].values])  # (L, 3)
                done_np    = df["done"].values.astype(np.float32)        # (L,)

                if has_phase:
                    phase_np = np.stack([np.array(v, dtype=np.float32)
                                         for v in df["phase"].values])  # (L, 2)
                else:
                    phase_np = np.zeros((len(df), 2), dtype=np.float32)

                if has_maneuver:
                    subgoal_np = df["subgoal_index"].values.astype(np.float32) / 2.0
                    cos_t_np   = df["cos_target"].values.astype(np.float32)
                    sin_t_np   = df["sin_target"].values.astype(np.float32)
                    herr_np    = df["heading_err"].values.astype(np.float32) / np.pi
                    lmp_np     = df["landmark_passed"].values.astype(np.float32)
                    maneuver_np = np.stack([subgoal_np, cos_t_np, sin_t_np, herr_np, lmp_np],
                                            axis=1)  # (L, 5)
                else:
                    maneuver_np = np.zeros((len(df), MANEUVER_DIM), dtype=np.float32)

                # Task description
                task_desc = str(df["task_description"].iloc[0])

                all_episodes.append({
                    "ep_id":       ep_id,
                    "length":      len(df),
                    "has_phase":   has_phase,
                    "has_maneuver": has_maneuver,
                    "repo":        str(repo_path),
                    "task_desc":   task_desc,
                    # Pre-converted numpy arrays for fast indexing
                    "proprio_np":  proprio_np,   # (L, 55)
                    "phase_np":    phase_np,     # (L, 2)
                    "maneuver_np": maneuver_np,  # (L, 5)
                    "action_np":   action_np,    # (L, 15)
                    "goal_np":     goal_np,      # (L, 3)
                    "vel_np":      vel_np,       # (L, 3)
                    "done_np":     done_np,      # (L,)
                })

        self._episodes = all_episodes

        # Build flat index: skip first K and last H frames
        self._index: list[tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            L = ep["length"]
            for t in range(self.K, L - self.H):
                self._index.append((i, t))

        n_maneuver = sum(1 for ep in all_episodes if ep["has_maneuver"])
        n_loco = len(all_episodes) - n_maneuver
        print(f"[ManeuverDataset] split={split}: {len(all_episodes)} episodes "
              f"({n_maneuver} maneuver, {n_loco} locomotion), "
              f"{len(self._index)} samples. proprio_dim={self.proprio_dim}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_i, t = self._index[idx]
        ep = self._episodes[ep_i]

        # Fast numpy array slicing — no pandas, no iterrows
        s = t - self.K
        e = t
        proprio_h = torch.from_numpy(
            np.concatenate([
                ep["proprio_np"][s:e],    # (K, 55)
                ep["phase_np"][s:e],      # (K, 2)
                ep["maneuver_np"][s:e],   # (K, 5)
            ], axis=1)                    # (K, 62)
        )

        # Action chunk
        action = torch.from_numpy(ep["action_np"][t:t + self.H])  # (H, 15)

        # Scalars at time t
        goal    = torch.from_numpy(ep["goal_np"][t])      # (3,)
        vel_cmd = torch.from_numpy(ep["vel_np"][t])       # (3,)
        done    = torch.tensor(ep["done_np"][t], dtype=torch.float32)

        # Ego RGB: zeros
        ego_rgb = torch.zeros(self.in_ch, self.img_size, self.img_size)

        # Lang emb
        task_desc = ep["task_desc"]
        if self._lang_cache is not None and task_desc in self._lang_cache:
            lang_emb = torch.from_numpy(self._lang_cache[task_desc].astype(np.float32))
        else:
            lang_emb = torch.zeros(2048)

        return dict(
            ego_rgb   = ego_rgb,
            proprio_h = proprio_h,   # (K, 62)
            lang_emb  = lang_emb,
            action    = action,
            goal      = goal,
            vel_cmd   = vel_cmd,
            done      = done,
        )


def make_maneuver_dataloader(
    repo_paths: list[str] | str,
    split: str = 'train',
    batch_size: int = 64,
    train_fraction: float = 0.9,
    num_workers: int = 0,
    lang_cache_path: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    ds = ManeuverParquetDataset(
        repo_paths      = repo_paths,
        split           = split,
        train_fraction  = train_fraction,
        lang_cache_path = lang_cache_path,
        **kwargs,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = (split == 'train'),
        num_workers = num_workers,
        pin_memory  = False,
    )
