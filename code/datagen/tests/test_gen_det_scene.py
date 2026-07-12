"""Integration smoke tests for code.datagen.gen_det_scene (RF-1).

make_scene_cfg is a thin dispatcher (cheap). run_scene is a real (but
cheap, ~2s) mujoco rollout+teleport capture.
"""

from __future__ import annotations

import unittest

import numpy as np

from code.datagen.gen_det_scene import make_scene_cfg, run_scene


class MakeSceneCfgTest(unittest.TestCase):
    def test_easy_dispatches_to_scene_sampler(self) -> None:
        rng = np.random.default_rng(0)
        cfg = make_scene_cfg(rng, "easy", ep_i=0)
        self.assertIn("objects", cfg)
        self.assertIn("target_index", cfg)
        self.assertIn("instruction", cfg)

    def test_demo_dispatches_to_scene_sampler(self) -> None:
        rng = np.random.default_rng(0)
        cfg = make_scene_cfg(rng, "demo", ep_i=0)
        self.assertIn("objects", cfg)

    def test_search_dispatches_to_search_sampler(self) -> None:
        rng = np.random.default_rng(0)
        cfg = make_scene_cfg(rng, "search", ep_i=0)
        self.assertIn("objects", cfg)
        self.assertIn("target_index", cfg)

    def test_deterministic_given_same_rng_state(self) -> None:
        cfg1 = make_scene_cfg(np.random.default_rng(42), "easy", ep_i=0)
        cfg2 = make_scene_cfg(np.random.default_rng(42), "easy", ep_i=0)
        self.assertEqual(cfg1["instruction"], cfg2["instruction"])
        self.assertEqual(len(cfg1["objects"]), len(cfg2["objects"]))


class RunSceneTest(unittest.TestCase):
    def test_easy_scene_produces_frame_records(self) -> None:
        rng_sample = np.random.default_rng(0)
        scene_cfg, frames = run_scene(scene_id=0, style="easy", base_seed=5,
                                      ep_i=0, rng_sample=rng_sample)
        self.assertIn("objects", scene_cfg)
        self.assertGreater(len(frames), 0)
        rec = frames[0]
        for key in ("rgb", "depth", "cam_type", "robot_x", "robot_y", "robot_yaw",
                    "qpos", "n_objects_visible", "labels", "source"):
            self.assertIn(key, rec)
        self.assertIn(rec["source"], ("trajectory", "teleport_focus", "teleport_random"))
        self.assertIn(rec["cam_type"], ("grounding", "proximity"))

    def test_frame_sources_include_trajectory_and_teleport(self) -> None:
        rng_sample = np.random.default_rng(1)
        _, frames = run_scene(scene_id=1, style="easy", base_seed=5,
                              ep_i=1, rng_sample=rng_sample)
        sources = {rec["source"] for rec in frames}
        # With default N_TELEPORT_FOCUS/N_TELEPORT_RANDOM > 0 we expect at
        # least trajectory + one teleport family to show up.
        self.assertIn("trajectory", sources)
        self.assertTrue(sources & {"teleport_focus", "teleport_random"})

    def test_same_scene_id_and_seed_is_deterministic(self) -> None:
        cfg1, frames1 = run_scene(scene_id=7, style="easy", base_seed=99,
                                  ep_i=7, rng_sample=np.random.default_rng(3))
        cfg2, frames2 = run_scene(scene_id=7, style="easy", base_seed=99,
                                  ep_i=7, rng_sample=np.random.default_rng(3))
        self.assertEqual(cfg1["instruction"], cfg2["instruction"])
        self.assertEqual(len(frames1), len(frames2))
        np.testing.assert_allclose(frames1[0]["qpos"], frames2[0]["qpos"], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
