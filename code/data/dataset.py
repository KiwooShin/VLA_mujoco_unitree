"""
code/data/dataset.py — Dataset for GroundedNav student.

Role: the base (55-d proprio) dataset layer. Three modes:
  1. SyntheticDataset  — schema-shaped random tensors; overfit-gate fallback.
  2. LeRobotDataset    — reads a LeRobot HDF5 repo on disk.
  3. ParquetDataset    — reads S3b parquet + mp4 video output (default for training).

ADR-001 schema (per timestep):
  ego_rgb      : (3 or 4, 128, 128)   uint8 → float [0,1] after normalize
  ego_depth    : (1, 128, 128)        float (merged into ego_rgb channel 4 if in_ch=4)
  proprio_h    : (K, 55)              K history frames
  lang_emb     : (2048,)              cached per-episode
  action       : (H, 15)             joint targets (chunked)
  goal         : (3,)                (dist, cosθ, sinθ)
  vel_cmd      : (3,)                (vx, vy, ωz)
  done         : scalar float         0/1

Split policy: by EPISODE (no within-episode leakage). Moved from
code/dataset.py (RF-1); phase/maneuver variants are duplicated (not
imported) in dataset_phase.py/dataset_maneuver.py, by design.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Synthetic Dataset (overfit-gate fallback)
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Returns schema-shaped random tensors. Used for overfit gate and CI.
    A small fixed set (n_samples) is generated at construction time so
    the model can overfit (same data every epoch).
    """

    def __init__(
        self,
        n_samples: int = 64,
        img_size: int = 128,
        in_ch: int = 3,
        proprio_dim: int = 55,
        proprio_K: int = 6,
        lang_dim: int = 2048,
        action_dim: int = 15,
        chunk_H: int = 1,
        goal_dim: int = 3,
        vel_dim: int = 3,
        seed: int = 42,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.n = n_samples
        rng = torch.Generator()
        rng.manual_seed(seed)

        def R(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=rng)

        # Pre-generate all samples on device (GPU-resident for speed)
        self.ego_rgb   = R(n_samples, in_ch, img_size, img_size)
        self.proprio_h = R(n_samples, proprio_K, proprio_dim)
        self.lang_emb  = R(n_samples, lang_dim)
        # Actions: random small targets near 0 (realistic for joints)
        self.action    = R(n_samples, chunk_H, action_dim) * 0.1
        # Goal: dist in [0.5, 5], cosθ/sinθ on unit circle
        angles = torch.rand(n_samples, generator=rng) * 2 * 3.14159
        self.goal = torch.stack([
            torch.rand(n_samples, generator=rng) * 4.5 + 0.5,
            angles.cos(),
            angles.sin(),
        ], dim=1)
        # Vel: small random
        self.vel_cmd = R(n_samples, vel_dim) * 0.3
        # Done: binary, ~10% positive
        self.done = (torch.rand(n_samples, generator=rng) < 0.1).float()

        if device != 'cpu':
            for attr in ['ego_rgb', 'proprio_h', 'lang_emb', 'action',
                         'goal', 'vel_cmd', 'done']:
                setattr(self, attr, getattr(self, attr).to(device))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return dict(
            ego_rgb   = self.ego_rgb[idx],
            proprio_h = self.proprio_h[idx],
            lang_emb  = self.lang_emb[idx],
            action    = self.action[idx],
            goal      = self.goal[idx],
            vel_cmd   = self.vel_cmd[idx],
            done      = self.done[idx],
        )


# ---------------------------------------------------------------------------
# LeRobot Dataset
# ---------------------------------------------------------------------------

class LeRobotDataset(Dataset):
    """
    Reads a LeRobot-format HDF5 repo (S3 schema).

    Expected directory structure (S3 will produce this):
      <repo_path>/
        data/
          chunk-000/
            episode_000000.hdf5
            ...
        meta/
          episodes.jsonl    # {episode_id, length, task_description, lang_emb_path}

    Each HDF5 has datasets: ego_rgb, ego_depth, proprio, action, goal, vel_cmd, done.
    lang_emb is stored per-episode (in meta or a separate npy file).

    Split: by episode_id. train_fraction of episodes → train; rest → val.
    """

    def __init__(
        self,
        repo_path: str,
        split: str = 'train',
        train_fraction: float = 0.9,
        proprio_K: int = 6,
        chunk_H: int = 1,
        img_size: int = 128,
        in_ch: int = 3,
        proprio_dim: int = 55,
    ) -> None:
        super().__init__()
        try:
            import h5py
            import json
            import numpy as np
        except ImportError as e:
            raise ImportError(f"LeRobotDataset requires h5py and numpy: {e}")

        self.repo = Path(repo_path)
        self.K = proprio_K
        self.H = chunk_H
        self.in_ch = in_ch
        self.proprio_dim = proprio_dim
        self.img_size = img_size

        meta_path = self.repo / 'meta' / 'episodes.jsonl'
        if not meta_path.exists():
            raise FileNotFoundError(f"meta/episodes.jsonl not found in {repo_path}")

        episodes = []
        with open(meta_path) as f:
            for line in f:
                episodes.append(json.loads(line.strip()))

        # Split by episode
        n_ep = len(episodes)
        n_train = max(1, int(n_ep * train_fraction))
        if split == 'train':
            episodes = episodes[:n_train]
        else:
            episodes = episodes[n_train:]

        # Build flat index: list of (episode_meta, t) pairs
        self._index = []  # (ep_meta, timestep_idx)
        for ep in episodes:
            length = ep['length']
            # Skip first K steps (no full history)
            for t in range(self.K, length - self.H):
                self._index.append((ep, t))

        self._episodes = episodes

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        import h5py
        import numpy as np

        ep_meta, t = self._index[idx]
        ep_id = ep_meta['episode_id']

        chunk_idx = ep_id // 1000
        hdf5_path = (self.repo / 'data' /
                     f'chunk-{chunk_idx:03d}' /
                     f'episode_{ep_id:06d}.hdf5')

        with h5py.File(hdf5_path, 'r') as f:
            # RGB: (T, H, W, C) uint8 → (C, H, W) float [0,1]
            rgb = f['ego_rgb'][t]  # (H, W, 3)
            rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

            if self.in_ch == 4:
                depth = f['ego_depth'][t]  # (H, W, 1) or (H, W)
                if depth.ndim == 2:
                    depth = depth[:, :, None]
                depth = torch.from_numpy(depth.astype(np.float32)).permute(2, 0, 1)
                ego_rgb = torch.cat([rgb, depth], dim=0)  # (4, H, W)
            else:
                ego_rgb = rgb  # (3, H, W)

            # Proprio history: K frames ending at t
            prop_slice = f['proprio'][t - self.K:t]  # (K, proprio_dim)
            proprio_h = torch.from_numpy(prop_slice.astype(np.float32))

            # Action chunk: H steps starting at t
            act_slice = f['action'][t:t + self.H]    # (H, 15)
            action = torch.from_numpy(act_slice.astype(np.float32))

            goal    = torch.from_numpy(f['goal'][t].astype(np.float32))
            vel_cmd = torch.from_numpy(f['vel_cmd'][t].astype(np.float32))
            done    = torch.tensor(float(f['done'][t]), dtype=torch.float32)

        # lang_emb: per episode (stored in meta or separate npy)
        lang_emb_path = ep_meta.get('lang_emb_path')
        if lang_emb_path and os.path.exists(lang_emb_path):
            import numpy as np
            lang_emb = torch.from_numpy(np.load(lang_emb_path).astype(np.float32))
        else:
            # Fallback: zero embedding (S3 should always provide this)
            lang_emb = torch.zeros(2048)

        return dict(
            ego_rgb   = ego_rgb,
            proprio_h = proprio_h,
            lang_emb  = lang_emb,
            action    = action,
            goal      = goal,
            vel_cmd   = vel_cmd,
            done      = done,
        )


# ---------------------------------------------------------------------------
# Parquet Dataset (S3b native format)
# ---------------------------------------------------------------------------

class ParquetDataset(Dataset):
    """
    Reads S3b-generated dataset (parquet per episode + mp4 videos).

    Directory structure:
      <repo_path>/
        data/chunk-000/
          episode_000000.parquet
          ...
        videos/
          episode_000000_ego.mp4
          ...
        meta/
          episodes.jsonl

    Each parquet row: frame_index, episode_index, proprio(55), action(15),
                      goal(3), vel_cmd(3), done, task_description.

    ego_rgb is decoded from the mp4 video on-the-fly (or a dummy zero tensor
    if video reading is disabled via load_video=False).

    lang_emb is looked up from a pre-built cache file
    (dataset/lang_cache.pkl built by code/groot_lang.py).
    If cache_path is None, lang_emb = zeros(2048).
    """

    def __init__(
        self,
        repo_path: str,
        split: str = 'train',
        train_fraction: float = 0.9,
        proprio_K: int = 6,
        chunk_H: int = 1,
        img_size: int = 128,
        in_ch: int = 3,
        proprio_dim: int = 55,
        lang_cache_path: str | None = None,
        load_video: bool = True,
    ) -> None:
        super().__init__()
        import json

        self.repo = Path(repo_path)
        self.K = proprio_K
        self.H = chunk_H
        self.in_ch = in_ch
        self.proprio_dim = proprio_dim
        self.img_size = img_size
        self.load_video = load_video
        self.lang_cache_path = lang_cache_path

        # Load lang cache if provided
        self._lang_cache: dict | None = None
        if lang_cache_path and os.path.exists(lang_cache_path):
            import pickle
            with open(lang_cache_path, "rb") as f:
                self._lang_cache = pickle.load(f)
            print(f"[ParquetDataset] Loaded lang cache: {len(self._lang_cache)} entries")
        elif lang_cache_path:
            print(f"[ParquetDataset] WARNING: lang cache not found at {lang_cache_path} — using zeros")

        # Discover all episode parquet files
        meta_path = self.repo / 'meta' / 'episodes.jsonl'
        episodes_meta = []
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes_meta.append(json.loads(line))

        # Fallback: scan chunk dirs
        if not episodes_meta:
            chunk_dir = self.repo / 'data' / 'chunk-000'
            if chunk_dir.exists():
                for p in sorted(chunk_dir.glob("episode_*.parquet")):
                    ep_id = int(p.stem.split("_")[1])
                    episodes_meta.append({"episode_index": ep_id})

        # Load all episode data frames into memory (indexed)
        self._episodes: list[dict] = []  # {ep_id, df, video_path}
        import pandas as pd
        for em in episodes_meta:
            ep_id = em.get("episode_index", em.get("episode_id", 0))
            chunk_idx = ep_id // 1000
            parq = self.repo / 'data' / f'chunk-{chunk_idx:03d}' / f'episode_{ep_id:06d}.parquet'
            if not parq.exists():
                continue
            df = pd.read_parquet(parq)
            vid = self.repo / 'videos' / f'episode_{ep_id:06d}_ego.mp4'
            self._episodes.append({
                "ep_id": ep_id,
                "df": df,
                "video_path": str(vid) if vid.exists() else None,
                "length": len(df),
            })

        # Split by episode
        n_ep = len(self._episodes)
        n_train = max(1, int(n_ep * train_fraction))
        if split == 'train':
            self._episodes = self._episodes[:n_train]
        else:
            self._episodes = self._episodes[n_train:]

        # Build flat index: (ep_idx, timestep_t)
        # Skip first K steps (no full proprio history), last H steps
        self._index: list[tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            L = ep["length"]
            for t in range(self.K, L - self.H):
                self._index.append((i, t))

        print(f"[ParquetDataset] split={split}: {len(self._episodes)} episodes, "
              f"{len(self._index)} samples")

    def __len__(self) -> int:
        return len(self._index)

    def _get_video_frame(self, video_path: str | None, t: int) -> torch.Tensor:
        """Decode frame t from mp4 → (3, img_size, img_size) float [0,1]."""
        if video_path is None or not self.load_video:
            return torch.zeros(self.in_ch, self.img_size, self.img_size)
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return torch.zeros(self.in_ch, self.img_size, self.img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            img = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            return img[:self.in_ch]
        except Exception:
            return torch.zeros(self.in_ch, self.img_size, self.img_size)

    def __getitem__(self, idx: int) -> dict:
        ep_i, t = self._index[idx]
        ep = self._episodes[ep_i]
        df = ep["df"]

        # Proprio history: K frames ending at t (exclusive)
        prop_rows = df.iloc[t - self.K:t]
        proprio_h = torch.from_numpy(
            np.stack([np.array(r, dtype=np.float32) for r in prop_rows['proprio']])
        )  # (K, 55)

        # Action chunk: H steps starting at t
        act_rows = df.iloc[t:t + self.H]
        action = torch.from_numpy(
            np.stack([np.array(r, dtype=np.float32) for r in act_rows['action']])
        )  # (H, 15)

        # Goal, vel_cmd, done at t
        goal    = torch.from_numpy(np.array(df['goal'].iloc[t],    dtype=np.float32))
        vel_cmd = torch.from_numpy(np.array(df['vel_cmd'].iloc[t], dtype=np.float32))
        done    = torch.tensor(float(df['done'].iloc[t]), dtype=torch.float32)

        # Ego RGB (from video)
        ego_rgb = self._get_video_frame(ep["video_path"], t)

        # Lang emb
        task_desc = str(df['task_description'].iloc[t])
        if self._lang_cache is not None and task_desc in self._lang_cache:
            lang_emb = torch.from_numpy(self._lang_cache[task_desc].astype(np.float32))
        else:
            lang_emb = torch.zeros(2048)

        return dict(
            ego_rgb   = ego_rgb,
            proprio_h = proprio_h,
            lang_emb  = lang_emb,
            action    = action,
            goal      = goal,
            vel_cmd   = vel_cmd,
            done      = done,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dataloader(
    mode: str = 'synthetic',       # 'synthetic', 'lerobot', or 'parquet'
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: str = 'cpu',           # for synthetic GPU-resident tensors
    repo_path: str | None = None,
    lang_cache_path: str | None = None,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for the given mode.

    For 'synthetic', tensors are pre-generated (overfit gate).
    For 'lerobot', reads from HDF5 disk.
    For 'parquet', reads from S3b parquet + mp4 format (default for real training).

    Args:
        mode: Dataset backend to use — 'synthetic', 'lerobot', or 'parquet'.
        split: 'train' or 'val' (controls shuffling and, for lerobot/parquet,
            the episode split).
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count (ignored for 'synthetic', which
            forces 0 since its tensors are GPU-resident).
        pin_memory: DataLoader pin_memory flag (forced False for 'synthetic').
        device: Device for synthetic GPU-resident tensors.
        repo_path: Dataset repo path, required for 'lerobot' and 'parquet' modes.
        lang_cache_path: Optional path to a pre-built language embedding cache
            (only used by 'parquet' mode).
        **dataset_kwargs: Forwarded to the underlying Dataset constructor.

    Returns:
        A configured DataLoader for the requested mode.

    Raises:
        ValueError: If `mode` is unknown, or if `repo_path` is None for the
            'lerobot' or 'parquet' modes.
    """
    if mode == 'synthetic':
        ds = SyntheticDataset(device=device, **dataset_kwargs)
        # For synthetic GPU-resident data, num_workers must be 0
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=0, pin_memory=False,
        )
    elif mode == 'lerobot':
        if repo_path is None:
            raise ValueError("repo_path required for lerobot mode")
        ds = LeRobotDataset(repo_path=repo_path, split=split, **dataset_kwargs)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=pin_memory,
        )
    elif mode == 'parquet':
        if repo_path is None:
            raise ValueError("repo_path required for parquet mode")
        ds = ParquetDataset(
            repo_path=repo_path, split=split,
            lang_cache_path=lang_cache_path,
            **dataset_kwargs,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")
