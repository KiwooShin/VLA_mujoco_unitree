"""Unit tests for code.policy.action_stats (RF-1).

Builds tiny synthetic LeRobot-style episode/parquet fixtures on disk (via
tempfile) to exercise compute_action_stats end to end (mean/std/n_frames,
the STD_FLOOR clamp, both episode-discovery paths, the JSON save/load round
trip), plus load_action_stats and stats_from_checkpoint's normal and error
paths.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from code.policy import action_stats as A


def _write_episode(repo: Path, ep_id: int, actions: np.ndarray) -> None:
    """Write one episode parquet under repo/data/chunk-000/episode_XXXXXX.parquet.

    Args:
        repo: Dataset repo root.
        ep_id: Episode index.
        actions: (N, 15) float array of per-frame actions.
    """
    chunk_dir = repo / 'data' / 'chunk-000'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'action': [row.tolist() for row in actions]})
    df.to_parquet(chunk_dir / f'episode_{ep_id:06d}.parquet')


def _write_meta(repo: Path, n_episodes: int) -> None:
    meta_dir = repo / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / 'episodes.jsonl', 'w') as f:
        for i in range(n_episodes):
            f.write(json.dumps({'episode_index': i}) + '\n')


class TestComputeActionStatsWithMeta(unittest.TestCase):
    """compute_action_stats via the meta/episodes.jsonl discovery path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_mean_std_match_hand_computed_values(self):
        rng = np.random.default_rng(0)
        # 2 episodes, deterministic per-joint deltas so we can hand-verify mean/std.
        n0, n1 = 10, 15
        base = A.DEFAULT_ANGLES
        deltas0 = rng.normal(0.0, 0.05, size=(n0, 15)).astype(np.float32)
        deltas1 = rng.normal(0.0, 0.05, size=(n1, 15)).astype(np.float32)
        _write_meta(self.repo, 2)
        _write_episode(self.repo, 0, base[np.newaxis, :] + deltas0)
        _write_episode(self.repo, 1, base[np.newaxis, :] + deltas1)

        stats = A.compute_action_stats(str(self.repo), train_fraction=1.0, verbose=False)

        all_deltas = np.concatenate([deltas0, deltas1], axis=0)
        expected_mean = all_deltas.mean(axis=0)
        expected_std = np.maximum(all_deltas.std(axis=0), A.STD_FLOOR)

        np.testing.assert_allclose(stats['mean'], expected_mean, atol=1e-4)
        np.testing.assert_allclose(stats['std'], expected_std, atol=1e-4)
        self.assertEqual(stats['n_frames'], n0 + n1)
        np.testing.assert_allclose(stats['default_angles'], base, atol=1e-6)
        self.assertEqual(stats['repo_path'], str(self.repo))

    def test_train_fraction_limits_episodes_scanned(self):
        rng = np.random.default_rng(1)
        for i in range(10):
            _write_episode(self.repo, i, A.DEFAULT_ANGLES[np.newaxis, :] +
                            rng.normal(0, 0.01, size=(5, 15)).astype(np.float32))
        _write_meta(self.repo, 10)

        stats = A.compute_action_stats(str(self.repo), train_fraction=0.3, verbose=False)
        # max(1, int(10*0.3)) == 3 episodes * 5 frames each
        self.assertEqual(stats['n_frames'], 15)

    def test_std_floor_applied_to_constant_joint(self):
        # waist_yaw-like joint with zero variance across all frames.
        n = 20
        actions = np.tile(A.DEFAULT_ANGLES, (n, 1)).astype(np.float32)
        _write_meta(self.repo, 1)
        _write_episode(self.repo, 0, actions)

        stats = A.compute_action_stats(str(self.repo), train_fraction=1.0, verbose=False)
        self.assertTrue(np.all(np.array(stats['std']) >= A.STD_FLOOR))
        self.assertTrue(np.allclose(stats['std'], A.STD_FLOOR))

    def test_stats_path_saves_json_and_is_loadable(self):
        _write_meta(self.repo, 1)
        _write_episode(self.repo, 0, A.DEFAULT_ANGLES[np.newaxis, :].repeat(4, axis=0))
        out_path = self.repo / 'stats.json'
        stats = A.compute_action_stats(str(self.repo), train_fraction=1.0,
                                        stats_path=str(out_path), verbose=False)
        self.assertTrue(out_path.exists())
        loaded = A.load_action_stats(str(out_path))
        np.testing.assert_allclose(loaded['mean'], stats['mean'], atol=1e-6)
        np.testing.assert_allclose(loaded['std'], stats['std'], atol=1e-6)
        np.testing.assert_allclose(loaded['default_angles'], stats['default_angles'], atol=1e-6)
        self.assertEqual(loaded['n_frames'], stats['n_frames'])
        self.assertIsInstance(loaded['mean'], np.ndarray)
        self.assertEqual(loaded['mean'].dtype, np.float32)


class TestComputeActionStatsWithoutMeta(unittest.TestCase):
    """compute_action_stats via the chunk-dir glob discovery path (no meta/)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_discovers_episodes_from_chunk_dir_glob(self):
        for i in range(3):
            _write_episode(self.repo, i,
                            A.DEFAULT_ANGLES[np.newaxis, :].repeat(2, axis=0))
        stats = A.compute_action_stats(str(self.repo), train_fraction=1.0, verbose=False)
        self.assertEqual(stats['n_frames'], 6)


class TestComputeActionStatsErrors(unittest.TestCase):
    """Error paths: missing repo entirely / no matching parquet files."""

    def test_no_episodes_found_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                A.compute_action_stats(d, verbose=False)

    def test_meta_present_but_parquet_missing_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            _write_meta(repo, 2)  # episodes.jsonl references ep 0,1 but no parquet written
            with self.assertRaises(RuntimeError):
                A.compute_action_stats(str(repo), train_fraction=1.0, verbose=False)


class TestLoadActionStats(unittest.TestCase):
    def test_round_trip_types(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'stats.json'
            raw = {
                'mean': [0.0] * 15,
                'std': [1.0] * 15,
                'default_angles': A.DEFAULT_ANGLES.tolist(),
                'n_frames': 42,
                'repo_path': 'some/path',
            }
            with open(path, 'w') as f:
                json.dump(raw, f)
            loaded = A.load_action_stats(str(path))
            self.assertEqual(loaded['mean'].shape, (15,))
            self.assertEqual(loaded['std'].dtype, np.float32)
            self.assertEqual(loaded['n_frames'], 42)
            self.assertEqual(loaded['repo_path'], 'some/path')

    def test_missing_repo_path_defaults_to_empty_string(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'stats.json'
            raw = {'mean': [0.0] * 15, 'std': [1.0] * 15,
                   'default_angles': A.DEFAULT_ANGLES.tolist(), 'n_frames': 1}
            with open(path, 'w') as f:
                json.dump(raw, f)
            loaded = A.load_action_stats(str(path))
            self.assertEqual(loaded['repo_path'], '')


class TestStatsFromCheckpoint(unittest.TestCase):
    def test_normal_path(self):
        ckpt = {'action_stats': {
            'mean': [0.1] * 15, 'std': [0.2] * 15,
            'default_angles': A.DEFAULT_ANGLES.tolist(), 'n_frames': 7,
        }}
        stats = A.stats_from_checkpoint(ckpt)
        self.assertEqual(stats['mean'].shape, (15,))
        self.assertEqual(stats['n_frames'], 7)
        self.assertIsInstance(stats['mean'], np.ndarray)

    def test_missing_action_stats_key_raises(self):
        with self.assertRaises(KeyError):
            A.stats_from_checkpoint({})

    def test_action_stats_present_but_no_mean_raises(self):
        with self.assertRaises(KeyError):
            A.stats_from_checkpoint({'action_stats': {'std': [1.0] * 15}})

    def test_n_frames_defaults_to_zero(self):
        ckpt = {'action_stats': {
            'mean': [0.0] * 15, 'std': [1.0] * 15,
            'default_angles': A.DEFAULT_ANGLES.tolist(),
        }}
        stats = A.stats_from_checkpoint(ckpt)
        self.assertEqual(stats['n_frames'], 0)


class TestDefaultAngles(unittest.TestCase):
    def test_shape_and_dtype(self):
        self.assertEqual(A.DEFAULT_ANGLES.shape, (15,))
        self.assertEqual(A.DEFAULT_ANGLES.dtype, np.float32)

    def test_left_right_leg_symmetry_of_defaults(self):
        # Left leg (0:6) and right leg (6:12) use the same default angles.
        np.testing.assert_allclose(A.DEFAULT_ANGLES[0:6], A.DEFAULT_ANGLES[6:12])

    def test_waist_defaults_are_zero(self):
        np.testing.assert_allclose(A.DEFAULT_ANGLES[12:15], 0.0)


if __name__ == "__main__":
    unittest.main()
