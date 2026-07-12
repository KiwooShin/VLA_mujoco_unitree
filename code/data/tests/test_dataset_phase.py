"""Unit tests for code.data.dataset_phase (RF-1): PhaseParquetDataset (57-d)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from code.data.dataset_phase import (
    PHASE_DIM,
    PROPRIO_DIM_BASE,
    PROPRIO_DIM_PHASE,
    PhaseParquetDataset,
    make_phase_dataloader,
)


def _write_episode(root: Path, ep_id: int, n_rows: int, with_phase: bool,
                   seed: int = 0, task: str = "go to the goal") -> None:
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
        if with_phase:
            phi = 2 * np.pi * t / n_rows
            row["phase"] = [float(np.sin(phi)), float(np.cos(phi))]
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


class PhaseParquetDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_constants(self) -> None:
        self.assertEqual(PROPRIO_DIM_BASE, 55)
        self.assertEqual(PHASE_DIM, 2)
        self.assertEqual(PROPRIO_DIM_PHASE, 57)

    def test_with_phase_column_appends_real_values(self) -> None:
        _write_episode(self.root, 0, n_rows=20, with_phase=True, seed=0)
        ds = PhaseParquetDataset(str(self.root), split="train", proprio_K=6, chunk_H=1)
        self.assertEqual(ds.proprio_dim, 57)
        item = ds[0]
        self.assertEqual(item["proprio_h"].shape, (6, 57))
        # phase columns (last 2 dims) should be within unit circle.
        phase = item["proprio_h"][:, 55:57]
        norms = (phase ** 2).sum(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_without_phase_column_zero_pads(self) -> None:
        _write_episode(self.root, 0, n_rows=20, with_phase=False, seed=0)
        ds = PhaseParquetDataset(str(self.root), split="train", proprio_K=6, chunk_H=1)
        item = ds[0]
        phase = item["proprio_h"][:, 55:57]
        self.assertTrue(torch.all(phase == 0))

    def test_multi_repo_combination(self) -> None:
        root2 = Path(tempfile.mkdtemp())
        try:
            _write_episode(self.root, 0, n_rows=20, with_phase=True, seed=1)
            _write_episode(root2, 0, n_rows=20, with_phase=False, seed=2)
            ds = PhaseParquetDataset([str(self.root), str(root2)], split="train",
                                     train_fraction=1.0)
            self.assertEqual(len(ds._episodes), 2)
        finally:
            import shutil
            shutil.rmtree(root2, ignore_errors=True)

    def test_split_is_per_repo(self) -> None:
        for ep in range(10):
            _write_episode(self.root, ep, n_rows=15, with_phase=True, seed=ep)
        train_ds = PhaseParquetDataset(str(self.root), split="train", train_fraction=0.8)
        val_ds = PhaseParquetDataset(str(self.root), split="val", train_fraction=0.8)
        self.assertEqual(len(train_ds._episodes), 8)
        self.assertEqual(len(val_ds._episodes), 2)

    def test_make_phase_dataloader_batches(self) -> None:
        _write_episode(self.root, 0, n_rows=20, with_phase=True, seed=0)
        dl = make_phase_dataloader(str(self.root), split="train", batch_size=2,
                                   train_fraction=1.0)
        batch = next(iter(dl))
        self.assertEqual(batch["proprio_h"].shape[1:], (6, 57))


if __name__ == "__main__":
    unittest.main()
