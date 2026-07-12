"""
code/data/dataset_phase.py — ParquetDataset extended to include gait phase [sin,cos] in proprio.

Role: the phase-aware (57-d proprio) dataset layer, used by train_dart_phase.py.

The phase-aware dataset reads the 'phase' column from each parquet row and appends
[sin(phi), cos(phi)] to the proprio vector, extending it from 55-d to 57-d.

This is used by train_dart_phase.py to train a phase-conditioned student model.

The model must be instantiated with proprio_dim=57 (not 55) when using this dataset.

Moved from code/dataset_phase.py (RF-1); see code/dataset_phase.py for the
old-path compat alias.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROPRIO_DIM_BASE  = 55   # original proprio without phase
PHASE_DIM         = 2    # [sin(phi), cos(phi)]
PROPRIO_DIM_PHASE = PROPRIO_DIM_BASE + PHASE_DIM   # 57


class PhaseParquetDataset(Dataset):
    """
    Reads S3b/DART parquet dataset (same format as ParquetDataset),
    but appends gait phase [sin_phi, cos_phi] to each proprio vector.

    If the 'phase' column is absent (clean dataset without phase),
    zeros are appended — the model sees [sin=0, cos=0] for clean data,
    which is a fixed point on the unit circle (phi=pi/2). This is slightly
    suboptimal but works — training then learns to mostly ignore phase
    when it's constant zero.

    Preferred: always generate/combine data WITH phase columns first.
    """

    def __init__(
        self,
        repo_paths: list[str] | str,    # one path or list of paths to combine
        split: str = 'train',
        train_fraction: float = 0.9,
        proprio_K: int = 6,
        chunk_H: int = 1,
        img_size: int = 128,
        in_ch: int = 3,
        lang_cache_path: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(repo_paths, str):
            repo_paths = [repo_paths]

        self.K = proprio_K
        self.H = chunk_H
        self.in_ch = in_ch
        self.img_size = img_size
        self.proprio_dim = PROPRIO_DIM_PHASE  # 57

        # Load lang cache
        self._lang_cache: dict | None = None
        if lang_cache_path and os.path.exists(lang_cache_path):
            import pickle
            with open(lang_cache_path, "rb") as f:
                self._lang_cache = pickle.load(f)
            print(f"[PhaseParquetDataset] Lang cache: {len(self._lang_cache)} entries")

        import pandas as pd

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
                # Fallback: scan parquet files
                chunk_dir = repo / "data" / "chunk-000"
                if chunk_dir.exists():
                    for p in sorted(chunk_dir.glob("episode_*.parquet")):
                        ep_id = int(p.stem.split("_")[1])
                        episodes_meta.append({"episode_index": ep_id})

            # Split by episode WITHIN this repo, then collect
            n_ep = len(episodes_meta)
            n_train = max(1, int(n_ep * train_fraction))
            if split == 'train':
                ep_slice = episodes_meta[:n_train]
            else:
                ep_slice = episodes_meta[n_train:]

            for em in ep_slice:
                ep_id   = em.get("episode_index", em.get("episode_id", 0))
                chunk_i = ep_id // 1000
                parq    = repo / "data" / f"chunk-{chunk_i:03d}" / f"episode_{ep_id:06d}.parquet"
                if not parq.exists():
                    continue
                df = pd.read_parquet(parq)
                has_phase = "phase" in df.columns
                all_episodes.append({
                    "ep_id":     ep_id,
                    "df":        df,
                    "has_phase": has_phase,
                    "length":    len(df),
                })

        self._episodes = all_episodes

        # Build flat index: skip first K and last H frames
        self._index: list[tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            L = ep["length"]
            for t in range(self.K, L - self.H):
                self._index.append((i, t))

        n_with_phase = sum(1 for ep in all_episodes if ep["has_phase"])
        print(f"[PhaseParquetDataset] split={split}: {len(all_episodes)} episodes "
              f"({n_with_phase} with phase column), {len(self._index)} samples. "
              f"proprio_dim={self.proprio_dim}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_i, t = self._index[idx]
        ep = self._episodes[ep_i]
        df = ep["df"]
        has_phase = ep["has_phase"]

        # Proprio history: K frames ending at t (exclusive)
        # Each frame: 55-d proprio + 2-d phase → 57-d total
        prop_rows = df.iloc[t - self.K:t]
        frames_57 = []
        for _, row in prop_rows.iterrows():
            p55 = np.array(row["proprio"], dtype=np.float32)  # (55,)
            if has_phase and "phase" in row.index:
                ph = np.array(row["phase"], dtype=np.float32)  # (2,)
            else:
                ph = np.zeros(2, dtype=np.float32)   # fallback: [sin=0, cos=0]
            frames_57.append(np.concatenate([p55, ph]))  # (57,)

        proprio_h = torch.from_numpy(np.stack(frames_57))   # (K, 57)

        # Action chunk: H steps starting at t
        act_rows = df.iloc[t:t + self.H]
        action = torch.from_numpy(
            np.stack([np.array(r, dtype=np.float32) for r in act_rows["action"]])
        )  # (H, 15)

        # Goal, vel_cmd, done at t
        goal    = torch.from_numpy(np.array(df["goal"].iloc[t],    dtype=np.float32))
        vel_cmd = torch.from_numpy(np.array(df["vel_cmd"].iloc[t], dtype=np.float32))
        done    = torch.tensor(float(df["done"].iloc[t]), dtype=torch.float32)

        # Ego RGB: always zeros (vision off)
        ego_rgb = torch.zeros(self.in_ch, self.img_size, self.img_size)

        # Lang emb
        task_desc = str(df["task_description"].iloc[t])
        if self._lang_cache is not None and task_desc in self._lang_cache:
            lang_emb = torch.from_numpy(self._lang_cache[task_desc].astype(np.float32))
        else:
            lang_emb = torch.zeros(2048)

        return dict(
            ego_rgb   = ego_rgb,
            proprio_h = proprio_h,   # (K, 57)
            lang_emb  = lang_emb,
            action    = action,
            goal      = goal,
            vel_cmd   = vel_cmd,
            done      = done,
        )


def make_phase_dataloader(
    repo_paths: list[str] | str,
    split: str = 'train',
    batch_size: int = 64,
    train_fraction: float = 0.9,
    num_workers: int = 0,
    lang_cache_path: str | None = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the phase-aware dataset.

    Args:
        repo_paths: One repo path or a list of paths to combine.
        split: 'train' or 'val'.
        batch_size: DataLoader batch size.
        train_fraction: Fraction of episodes (per repo) used for training.
        num_workers: DataLoader worker count.
        lang_cache_path: Optional path to a pre-built language embedding cache.
        **kwargs: Forwarded to `PhaseParquetDataset`.

    Returns:
        A configured DataLoader over `PhaseParquetDataset`.
    """
    ds = PhaseParquetDataset(
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
