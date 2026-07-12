"""Integration smoke tests for code.datagen.gen_dart_rollout.run_dart_episode.

Real mujoco + WBCTeacher + a real 'easy' scene, but with a tiny hard_maxsteps
so each run costs well under a second.
"""

from __future__ import annotations

import unittest

import numpy as np

from code.datagen.gen_dart_rollout import FALL_HEIGHT, FPS, HOLD_STEPS, run_dart_episode
from code.scene import derive_rng, sample_scene
from code.teacher import WBCTeacher


class RunDartEpisodeTest(unittest.TestCase):
    def test_short_episode_returns_well_formed_rows(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=7, episode_idx=0)
        scene_cfg = sample_scene(rng, "easy")

        result = run_dart_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=0,
            global_frame_offset=0, noise_sigma=0.07, hard_maxsteps=25,
            rng_noise=np.random.default_rng(1),
        )

        self.assertIsNotNone(result)
        rows = result["rows"]
        self.assertGreater(len(rows), 0)
        self.assertLessEqual(len(rows), 25)
        self.assertEqual(result["n_steps"], len(rows))

        first = rows[0]
        self.assertEqual(len(first["proprio"]), 55)
        self.assertEqual(len(first["action"]), 15)
        self.assertEqual(len(first["goal"]), 3)
        self.assertEqual(len(first["vel_cmd"]), 3)
        self.assertEqual(len(first["phase"]), 2)
        self.assertIn("task_description", first)
        self.assertEqual(first["episode_index"], 0)
        # timestamps derived from FPS
        self.assertAlmostEqual(rows[1]["timestamp"] - rows[0]["timestamp"], 1.0 / FPS, places=6)

    def test_global_frame_offset_applied(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=7, episode_idx=0)
        scene_cfg = sample_scene(rng, "easy")
        result = run_dart_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=3,
            global_frame_offset=1000, noise_sigma=0.0, hard_maxsteps=10,
            rng_noise=np.random.default_rng(2),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["rows"][0]["index"], 1000)
        self.assertEqual(result["rows"][0]["episode_index"], 3)

    def test_zero_noise_is_accepted(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=1, episode_idx=1)
        scene_cfg = sample_scene(rng, "easy")
        result = run_dart_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=0,
            global_frame_offset=0, noise_sigma=0.0, hard_maxsteps=10,
            rng_noise=np.random.default_rng(0),
        )
        self.assertIsNotNone(result)

    def test_default_rng_noise_when_none(self) -> None:
        teacher = WBCTeacher()
        rng = derive_rng(base_seed=2, episode_idx=0)
        scene_cfg = sample_scene(rng, "easy")
        result = run_dart_episode(
            teacher=teacher, scene_cfg=scene_cfg, episode_idx=0,
            global_frame_offset=0, noise_sigma=0.05, hard_maxsteps=10,
            rng_noise=None,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
