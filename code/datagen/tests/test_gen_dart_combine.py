"""Unit tests for code.datagen.gen_dart_combine (RF-1): add-phase / combine.

Pure pandas/parquet manipulation -- no mujoco physics needed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from code.datagen.gen_dart_combine import add_phase_to_clean_dataset, combine_datasets


def _make_clean_episode_df(n_rows: int, seed: int, task: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_rows):
        rows.append({
            "frame_index": t,
            "episode_index": 0,
            "index": t,
            "task_index": 0,
            "timestamp": t / 50.0,
            "proprio": rng.standard_normal(55).astype(np.float32).tolist(),
            "action": rng.standard_normal(15).astype(np.float32).tolist(),
            "goal": [float(n_rows - t) / 10.0, 1.0, 0.0],
            "vel_cmd": rng.standard_normal(3).astype(np.float32).tolist(),
            "done": int(t == n_rows - 1),
            "task_description": task,
        })
    return pd.DataFrame(rows)


def _write_clean_dataset(root: Path, n_episodes: int, n_rows: int) -> None:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    for i in range(n_episodes):
        df = _make_clean_episode_df(n_rows, seed=i, task=f"go to object {i}")
        df.to_parquet(data_dir / f"episode_{i:06d}.parquet", index=False)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.0"}))
    (root / "meta" / "episodes.jsonl").write_text("")


def _write_dart_dataset(root: Path, n_episodes: int, n_rows: int, with_phase: bool) -> None:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_episodes):
        df = _make_clean_episode_df(n_rows, seed=100 + i, task=f"dart task {i}")
        if with_phase:
            phis = np.linspace(0, 2 * np.pi, n_rows, endpoint=False)
            df["phase"] = [[float(np.sin(p)), float(np.cos(p))] for p in phis]
        df.to_parquet(data_dir / f"episode_{i:06d}.parquet", index=False)


class AddPhaseToCleanDatasetTest(unittest.TestCase):
    def test_adds_phase_column_and_copies_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "clean"
            out_dir = Path(tmp) / "clean_with_phase"
            _write_clean_dataset(in_dir, n_episodes=2, n_rows=20)

            total = add_phase_to_clean_dataset(str(in_dir), str(out_dir))

            self.assertEqual(total, 40)
            self.assertTrue((out_dir / "meta" / "info.json").exists())
            df0 = pd.read_parquet(out_dir / "data" / "chunk-000" / "episode_000000.parquet")
            self.assertIn("phase", df0.columns)
            self.assertEqual(len(df0.iloc[0]["phase"]), 2)
            # unit-circle invariant
            norms = df0["phase"].apply(lambda p: p[0] ** 2 + p[1] ** 2)
            np.testing.assert_allclose(norms.to_numpy(), np.ones(len(df0)), atol=1e-5)

    def test_empty_input_dir_gives_zero_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "clean"
            out_dir = Path(tmp) / "out"
            (in_dir / "data" / "chunk-000").mkdir(parents=True)
            (in_dir / "meta").mkdir(parents=True)
            total = add_phase_to_clean_dataset(str(in_dir), str(out_dir))
            self.assertEqual(total, 0)


class CombineDatasetsTest(unittest.TestCase):
    def test_combine_reindexes_and_flags_dart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_dir = Path(tmp) / "clean_with_phase"
            dart_dir = Path(tmp) / "dart"
            out_dir = Path(tmp) / "combined"
            _write_dart_dataset(clean_dir, n_episodes=2, n_rows=15, with_phase=True)
            _write_dart_dataset(dart_dir, n_episodes=3, n_rows=10, with_phase=True)

            info = combine_datasets(str(clean_dir), str(dart_dir), str(out_dir))

            self.assertEqual(info["n_clean_episodes"], 2)
            self.assertEqual(info["n_dart_episodes"], 3)
            self.assertEqual(info["total_episodes"], 5)
            self.assertEqual(info["total_frames"], 2 * 15 + 3 * 10)

            episodes_meta = [
                json.loads(l) for l in
                (out_dir / "meta" / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(episodes_meta), 5)
            # First n_clean episodes are NOT dart; the rest are.
            is_dart_flags = [em["is_dart"] for em in episodes_meta]
            self.assertEqual(is_dart_flags, [False, False, True, True, True])
            # episode_index / index / frame_index were re-numbered contiguously.
            self.assertEqual([em["episode_index"] for em in episodes_meta], [0, 1, 2, 3, 4])

            # Files actually exist on disk with the new numbering.
            for i in range(5):
                self.assertTrue((out_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet").exists())

            stats = json.loads((out_dir / "meta" / "stats.json").read_text())
            self.assertIn("mean", stats["proprio"])
            self.assertEqual(len(stats["proprio"]["mean"]), 55)
            self.assertEqual(len(stats["action"]["mean"]), 15)

    def test_combine_computes_phase_when_missing_from_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_dir = Path(tmp) / "clean_no_phase"
            dart_dir = Path(tmp) / "dart"
            out_dir = Path(tmp) / "combined"
            _write_dart_dataset(clean_dir, n_episodes=1, n_rows=20, with_phase=False)
            _write_dart_dataset(dart_dir, n_episodes=1, n_rows=10, with_phase=True)

            combine_datasets(str(clean_dir), str(dart_dir), str(out_dir))

            df0 = pd.read_parquet(out_dir / "data" / "chunk-000" / "episode_000000.parquet")
            self.assertIn("phase", df0.columns)

    def test_task_index_assignment_is_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_dir = Path(tmp) / "clean"
            dart_dir = Path(tmp) / "dart"
            out_dir = Path(tmp) / "combined"
            _write_dart_dataset(clean_dir, n_episodes=2, n_rows=5, with_phase=True)
            _write_dart_dataset(dart_dir, n_episodes=2, n_rows=5, with_phase=True)

            combine_datasets(str(clean_dir), str(dart_dir), str(out_dir))

            tasks = [json.loads(l) for l in
                    (out_dir / "meta" / "tasks.jsonl").read_text().splitlines()]
            task_indices = {t["task_index"] for t in tasks}
            self.assertEqual(task_indices, set(range(len(tasks))))

            for i in range(4):
                df = pd.read_parquet(out_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
                task = df["task_description"].iloc[0]
                idx = df["task_index"].iloc[0]
                matching = [t for t in tasks if t["task"] == task]
                self.assertEqual(matching[0]["task_index"], idx)


if __name__ == "__main__":
    unittest.main()
