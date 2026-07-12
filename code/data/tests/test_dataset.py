"""Unit tests for code.data.dataset (RF-1).

Covers SyntheticDataset, ParquetDataset (built against synthetic on-disk
parquet fixtures — no real dataset needed), LeRobotDataset's guarded h5py
dependency, and the make_dataloader factory's dispatch/error paths.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from code.data.dataset import (
    LeRobotDataset,
    ParquetDataset,
    SyntheticDataset,
    make_dataloader,
)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _write_episode_parquet(root: Path, ep_id: int, n_rows: int, seed: int = 0,
                           task: str = "go to the blue ball") -> None:
    """Writes one synthetic episode parquet (+ meta) under `root`."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_rows):
        rows.append({
            "frame_index": t,
            "episode_index": ep_id,
            "proprio": rng.standard_normal(55).astype(np.float32).tolist(),
            "action": rng.standard_normal(15).astype(np.float32).tolist(),
            "goal": rng.standard_normal(3).astype(np.float32).tolist(),
            "vel_cmd": rng.standard_normal(3).astype(np.float32).tolist(),
            "done": int(t == n_rows - 1),
            "task_description": task,
        })
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{ep_id:06d}.parquet", index=False)


def _write_meta(root: Path, ep_ids: list[int]) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_id in ep_ids:
            f.write(json.dumps({"episode_index": ep_id}) + "\n")


class SyntheticDatasetTest(unittest.TestCase):
    def test_default_shapes(self) -> None:
        ds = SyntheticDataset(n_samples=8)
        self.assertEqual(len(ds), 8)
        item = ds[0]
        self.assertEqual(item["ego_rgb"].shape, (3, 128, 128))
        self.assertEqual(item["proprio_h"].shape, (6, 55))
        self.assertEqual(item["lang_emb"].shape, (2048,))
        self.assertEqual(item["action"].shape, (1, 15))
        self.assertEqual(item["goal"].shape, (3,))
        self.assertEqual(item["vel_cmd"].shape, (3,))
        self.assertEqual(item["done"].shape, ())

    def test_custom_shapes(self) -> None:
        ds = SyntheticDataset(
            n_samples=4, img_size=64, in_ch=4, proprio_dim=60, proprio_K=8,
            lang_dim=100, action_dim=15, chunk_H=3,
        )
        item = ds[0]
        self.assertEqual(item["ego_rgb"].shape, (4, 64, 64))
        self.assertEqual(item["proprio_h"].shape, (8, 60))
        self.assertEqual(item["lang_emb"].shape, (100,))
        self.assertEqual(item["action"].shape, (3, 15))

    def test_goal_vector_invariants(self) -> None:
        ds = SyntheticDataset(n_samples=200, seed=7)
        dist = ds.goal[:, 0]
        cos_th = ds.goal[:, 1]
        sin_th = ds.goal[:, 2]
        self.assertTrue(torch.all(dist >= 0.5))
        self.assertTrue(torch.all(dist <= 5.0))
        unit = cos_th ** 2 + sin_th ** 2
        self.assertTrue(torch.allclose(unit, torch.ones_like(unit), atol=1e-5))

    def test_done_is_binary_and_roughly_ten_percent(self) -> None:
        ds = SyntheticDataset(n_samples=2000, seed=1)
        self.assertTrue(set(torch.unique(ds.done).tolist()) <= {0.0, 1.0})
        rate = ds.done.mean().item()
        # p=0.1 true rate; n=2000 gives std ~0.0067 -- generous bounds to avoid flakes.
        self.assertGreater(rate, 0.03)
        self.assertLess(rate, 0.20)

    def test_seed_determinism(self) -> None:
        a = SyntheticDataset(n_samples=16, seed=123)
        b = SyntheticDataset(n_samples=16, seed=123)
        self.assertTrue(torch.equal(a.ego_rgb, b.ego_rgb))
        self.assertTrue(torch.equal(a.proprio_h, b.proprio_h))
        self.assertTrue(torch.equal(a.action, b.action))
        self.assertTrue(torch.equal(a.goal, b.goal))

    def test_different_seed_differs(self) -> None:
        a = SyntheticDataset(n_samples=16, seed=1)
        b = SyntheticDataset(n_samples=16, seed=2)
        self.assertFalse(torch.equal(a.ego_rgb, b.ego_rgb))

    def test_len_matches_n_samples(self) -> None:
        ds = SyntheticDataset(n_samples=37)
        self.assertEqual(len(ds), 37)


class ParquetDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_basic_getitem_shapes_no_video(self) -> None:
        _write_episode_parquet(self.root, 0, n_rows=20, seed=0)
        _write_meta(self.root, [0])
        ds = ParquetDataset(str(self.root), split="train", proprio_K=6, chunk_H=1)
        # 20 rows, K=6, H=1 -> index range(6, 19) -> 13 samples
        self.assertEqual(len(ds), 13)
        item = ds[0]
        self.assertEqual(item["proprio_h"].shape, (6, 55))
        self.assertEqual(item["action"].shape, (1, 15))
        self.assertEqual(item["goal"].shape, (3,))
        self.assertEqual(item["vel_cmd"].shape, (3,))
        self.assertEqual(item["ego_rgb"].shape, (3, 128, 128))
        # No video on disk -> zero frame.
        self.assertTrue(torch.all(item["ego_rgb"] == 0))
        # No lang cache -> zeros.
        self.assertTrue(torch.all(item["lang_emb"] == 0))

    def test_load_video_false_forces_zero_frame(self) -> None:
        _write_episode_parquet(self.root, 0, n_rows=10, seed=0)
        _write_meta(self.root, [0])
        ds = ParquetDataset(str(self.root), split="train", load_video=False)
        item = ds[0]
        self.assertTrue(torch.all(item["ego_rgb"] == 0))

    def test_lang_cache_lookup(self) -> None:
        task = "please fetch the red cone"
        _write_episode_parquet(self.root, 0, n_rows=10, seed=0, task=task)
        _write_meta(self.root, [0])
        cache_path = self.root / "lang_cache.pkl"
        emb = np.ones(2048, dtype=np.float32) * 3.0
        with open(cache_path, "wb") as f:
            pickle.dump({task: emb}, f)
        ds = ParquetDataset(str(self.root), split="train", lang_cache_path=str(cache_path))
        item = ds[0]
        self.assertTrue(torch.allclose(item["lang_emb"], torch.from_numpy(emb)))

    def test_lang_cache_missing_task_falls_back_to_zeros(self) -> None:
        _write_episode_parquet(self.root, 0, n_rows=10, seed=0, task="task A")
        _write_meta(self.root, [0])
        cache_path = self.root / "lang_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump({"some other task": np.ones(2048, dtype=np.float32)}, f)
        ds = ParquetDataset(str(self.root), split="train", lang_cache_path=str(cache_path))
        item = ds[0]
        self.assertTrue(torch.all(item["lang_emb"] == 0))

    def test_train_val_split_by_episode(self) -> None:
        for ep in range(10):
            _write_episode_parquet(self.root, ep, n_rows=10, seed=ep)
        _write_meta(self.root, list(range(10)))
        train_ds = ParquetDataset(str(self.root), split="train", train_fraction=0.8)
        val_ds = ParquetDataset(str(self.root), split="val", train_fraction=0.8)
        self.assertEqual(len(train_ds._episodes), 8)
        self.assertEqual(len(val_ds._episodes), 2)

    def test_fallback_scan_without_meta(self) -> None:
        # No meta/episodes.jsonl: falls back to scanning data/chunk-000/*.parquet
        _write_episode_parquet(self.root, 0, n_rows=10, seed=0)
        _write_episode_parquet(self.root, 1, n_rows=10, seed=1)
        ds = ParquetDataset(str(self.root), split="train", train_fraction=1.0)
        self.assertEqual(len(ds._episodes), 2)

    def test_missing_parquet_file_is_skipped(self) -> None:
        # meta references an episode with no matching parquet on disk.
        _write_episode_parquet(self.root, 0, n_rows=10, seed=0)
        _write_meta(self.root, [0, 99])
        ds = ParquetDataset(str(self.root), split="train", train_fraction=1.0)
        self.assertEqual(len(ds._episodes), 1)


class MakeDataloaderTest(unittest.TestCase):
    def test_synthetic_mode_forces_single_process(self) -> None:
        dl = make_dataloader(mode="synthetic", batch_size=4, num_workers=4,
                              pin_memory=True, n_samples=16)
        self.assertEqual(dl.num_workers, 0)
        self.assertFalse(dl.pin_memory)
        batch = next(iter(dl))
        self.assertEqual(batch["ego_rgb"].shape[0], 4)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_dataloader(mode="bogus")

    def test_lerobot_requires_repo_path(self) -> None:
        with self.assertRaises(ValueError):
            make_dataloader(mode="lerobot", repo_path=None)

    def test_parquet_requires_repo_path(self) -> None:
        with self.assertRaises(ValueError):
            make_dataloader(mode="parquet", repo_path=None)

    def test_parquet_mode_builds_working_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episode_parquet(root, 0, n_rows=20, seed=0)
            _write_meta(root, [0])
            dl = make_dataloader(mode="parquet", repo_path=str(root), batch_size=2,
                                 split="train")
            batch = next(iter(dl))
            self.assertEqual(batch["proprio_h"].shape[1:], (6, 55))


class LeRobotDatasetTest(unittest.TestCase):
    """LeRobotDataset requires h5py; test both the guarded-import failure
    path (this env) and, when available, the real HDF5 read path."""

    def test_missing_meta_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            if not HAS_H5PY:
                with self.assertRaises(ImportError):
                    LeRobotDataset(repo_path=tmp)
            else:
                with self.assertRaises(FileNotFoundError):
                    LeRobotDataset(repo_path=tmp)

    @unittest.skipUnless(HAS_H5PY, "h5py not installed in this environment")
    def test_hdf5_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            n_rows = 20
            with h5py.File(root / "data" / "chunk-000" / "episode_000000.hdf5", "w") as f:
                f.create_dataset("ego_rgb", data=np.zeros((n_rows, 128, 128, 3), dtype=np.uint8))
                f.create_dataset("proprio", data=np.zeros((n_rows, 55), dtype=np.float32))
                f.create_dataset("action", data=np.zeros((n_rows, 15), dtype=np.float32))
                f.create_dataset("goal", data=np.zeros((n_rows, 3), dtype=np.float32))
                f.create_dataset("vel_cmd", data=np.zeros((n_rows, 3), dtype=np.float32))
                f.create_dataset("done", data=np.zeros((n_rows,), dtype=np.float32))
            with open(root / "meta" / "episodes.jsonl", "w") as f:
                f.write(json.dumps({"episode_id": 0, "length": n_rows,
                                    "task_description": "x"}) + "\n")
            ds = LeRobotDataset(repo_path=str(root), split="train", proprio_K=6, chunk_H=1)
            self.assertEqual(len(ds), n_rows - 6 - 1)
            item = ds[0]
            self.assertEqual(item["proprio_h"].shape, (6, 55))
            self.assertEqual(item["ego_rgb"].shape, (3, 128, 128))


if __name__ == "__main__":
    unittest.main()
