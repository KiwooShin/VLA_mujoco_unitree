"""Unit tests for code.data.dataset_maneuver (RF-1): ManeuverParquetDataset (62-d)."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from code.data.dataset_maneuver import (
    MANEUVER_DIM,
    PROPRIO_DIM_BASE,
    PROPRIO_DIM_MANEUVER,
    PROPRIO_DIM_PHASE,
    ManeuverParquetDataset,
    _build_maneuver_features,
    make_maneuver_dataloader,
)


class BuildManeuverFeaturesTest(unittest.TestCase):
    def test_full_row(self) -> None:
        row = {
            "subgoal_index": 1,
            "cos_target": 0.5,
            "sin_target": 0.8660254,
            "heading_err": math.pi / 2,
            "landmark_passed": 1,
        }
        feat = _build_maneuver_features(row)
        self.assertEqual(feat.shape, (5,))
        np.testing.assert_allclose(feat, [0.5, 0.5, 0.8660254, 0.5, 1.0], atol=1e-5)
        self.assertEqual(feat.dtype, np.float32)

    def test_defaults_on_empty_row(self) -> None:
        feat = _build_maneuver_features({})
        np.testing.assert_allclose(feat, [0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_subgoal_index_normalization(self) -> None:
        for idx, expected in [(0, 0.0), (1, 0.5), (2, 1.0)]:
            feat = _build_maneuver_features({"subgoal_index": idx})
            self.assertAlmostEqual(float(feat[0]), expected, places=6)

    def test_negative_heading_err(self) -> None:
        feat = _build_maneuver_features({"heading_err": -math.pi})
        self.assertAlmostEqual(float(feat[3]), -1.0, places=5)

    def test_landmark_passed_bool_like(self) -> None:
        feat = _build_maneuver_features({"landmark_passed": True})
        self.assertEqual(float(feat[4]), 1.0)


def _write_episode(root: Path, ep_id: int, n_rows: int, maneuver: bool,
                   phase: bool = True, seed: int = 0,
                   task: str = "pass the red cube and turn left") -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_rows):
        row = {
            "frame_index": t,
            "episode_index": ep_id,
            "proprio": rng.standard_normal(55).astype(np.float32).tolist(),
            "action": rng.standard_normal(15).astype(np.float32).tolist(),
            "goal": rng.standard_normal(3).astype(np.float32).tolist(),
            "vel_cmd": rng.standard_normal(3).astype(np.float32).tolist(),
            "done": int(t == n_rows - 1),
            "task_description": task,
        }
        if phase:
            phi = 2 * np.pi * t / n_rows
            row["phase"] = [float(np.sin(phi)), float(np.cos(phi))]
        if maneuver:
            row["subgoal_index"] = 1
            row["cos_target"] = 0.5
            row["sin_target"] = 0.866
            row["heading_err"] = 0.1 * t
            row["landmark_passed"] = int(t > n_rows // 2)
        rows.append(row)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{ep_id:06d}.parquet", index=False)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "episodes.jsonl"
    existing = []
    if meta_path.exists():
        existing = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    existing.append({"episode_index": ep_id})
    with open(meta_path, "w") as f:
        for em in existing:
            f.write(json.dumps(em) + "\n")


class ManeuverParquetDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_constants(self) -> None:
        self.assertEqual(PROPRIO_DIM_BASE, 55)
        self.assertEqual(PROPRIO_DIM_PHASE, 57)
        self.assertEqual(MANEUVER_DIM, 5)
        self.assertEqual(PROPRIO_DIM_MANEUVER, 62)

    def test_maneuver_episode_shapes(self) -> None:
        _write_episode(self.root, 0, n_rows=20, maneuver=True, seed=0)
        ds = ManeuverParquetDataset(str(self.root), split="train", proprio_K=6, chunk_H=1)
        self.assertEqual(ds.proprio_dim, 62)
        item = ds[0]
        self.assertEqual(item["proprio_h"].shape, (6, 62))
        maneuver_slice = item["proprio_h"][:, 57:62]
        self.assertTrue(torch.any(maneuver_slice != 0))
        self.assertEqual(ds._episodes[0]["has_maneuver"], True)

    def test_locomotion_episode_zero_pads_maneuver_dims(self) -> None:
        _write_episode(self.root, 0, n_rows=20, maneuver=False, phase=True, seed=0)
        ds = ManeuverParquetDataset(str(self.root), split="train", proprio_K=6, chunk_H=1)
        item = ds[0]
        maneuver_slice = item["proprio_h"][:, 57:62]
        self.assertTrue(torch.all(maneuver_slice == 0))
        self.assertEqual(ds._episodes[0]["has_maneuver"], False)

    def test_mixed_maneuver_and_locomotion(self) -> None:
        root2 = Path(tempfile.mkdtemp())
        try:
            _write_episode(self.root, 0, n_rows=20, maneuver=True, seed=1)
            _write_episode(root2, 0, n_rows=20, maneuver=False, seed=2)
            ds = ManeuverParquetDataset([str(self.root), str(root2)], split="train",
                                        train_fraction=1.0)
            self.assertEqual(len(ds._episodes), 2)
            has_maneuver_flags = sorted(ep["has_maneuver"] for ep in ds._episodes)
            self.assertEqual(has_maneuver_flags, [False, True])
        finally:
            import shutil
            shutil.rmtree(root2, ignore_errors=True)

    def test_no_phase_no_maneuver_all_zero_tail(self) -> None:
        _write_episode(self.root, 0, n_rows=20, maneuver=False, phase=False, seed=0)
        ds = ManeuverParquetDataset(str(self.root), split="train")
        item = ds[0]
        self.assertTrue(torch.all(item["proprio_h"][:, 55:] == 0))

    def test_make_maneuver_dataloader_batches(self) -> None:
        _write_episode(self.root, 0, n_rows=20, maneuver=True, seed=0)
        dl = make_maneuver_dataloader(str(self.root), split="train", batch_size=2,
                                      train_fraction=1.0)
        batch = next(iter(dl))
        self.assertEqual(batch["proprio_h"].shape[1:], (6, 62))
        self.assertEqual(batch["action"].shape[1:], (1, 15))


if __name__ == "__main__":
    unittest.main()
