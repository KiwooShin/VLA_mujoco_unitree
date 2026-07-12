"""Integration smoke tests for code.datagen.gen_det_capture (RF-1).

Real (cheap) mujoco render: build one 'easy' scene arena, capture one frame
per camera type.
"""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from code.arena import ArenaRenderer, build_arena
from code.datagen.gen_det_capture import SegRenderer, capture_frame
from code.datagen.gen_det_labels import build_id_to_obj
from code.scene import derive_rng, sample_scene


class CaptureFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = derive_rng(base_seed=11, episode_idx=0)
        self.scene_cfg = sample_scene(rng, "easy")
        self.model = build_arena(self.scene_cfg)
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.objects = self.scene_cfg["objects"]
        self.id_to_obj = build_id_to_obj(self.model, len(self.objects))
        self.renderer = ArenaRenderer(self.model)
        self.seg_rend = SegRenderer(self.model)

    def tearDown(self) -> None:
        self.renderer.close()
        self.seg_rend.close()

    def test_capture_grounding_frame(self) -> None:
        rec = capture_frame(self.renderer, self.seg_rend, self.data, yaw=0.0,
                            cam_type="grounding", objects=self.objects,
                            id_to_obj=self.id_to_obj)
        self.assertEqual(rec["cam_type"], "grounding")
        self.assertEqual(rec["rgb"].ndim, 3)
        self.assertEqual(rec["rgb"].shape[2], 3)
        self.assertEqual(rec["depth"].dtype, np.float16)
        self.assertEqual(rec["qpos"].shape, (self.model.nq,))
        self.assertIsInstance(rec["labels"], list)
        self.assertEqual(rec["n_objects_visible"], len(rec["labels"]))

    def test_capture_proximity_frame(self) -> None:
        rec = capture_frame(self.renderer, self.seg_rend, self.data, yaw=0.0,
                            cam_type="proximity", objects=self.objects,
                            id_to_obj=self.id_to_obj)
        self.assertEqual(rec["cam_type"], "proximity")
        self.assertEqual(rec["rgb"].shape[2], 3)

    def test_robot_pose_reflected_in_record(self) -> None:
        self.data.qpos[0] = 1.5
        self.data.qpos[1] = -0.5
        mujoco.mj_forward(self.model, self.data)
        rec = capture_frame(self.renderer, self.seg_rend, self.data, yaw=0.3,
                            cam_type="grounding", objects=self.objects,
                            id_to_obj=self.id_to_obj)
        self.assertAlmostEqual(rec["robot_x"], 1.5, places=5)
        self.assertAlmostEqual(rec["robot_y"], -0.5, places=5)


class SegRendererTest(unittest.TestCase):
    def test_render_returns_two_channel_int_array(self) -> None:
        rng = derive_rng(base_seed=12, episode_idx=0)
        scene_cfg = sample_scene(rng, "easy")
        model = build_arena(scene_cfg)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        seg_rend = SegRenderer(model)
        try:
            seg = seg_rend.render(data, yaw=0.0, cam_type="grounding")
            self.assertEqual(seg.ndim, 3)
            self.assertEqual(seg.shape[2], 2)
        finally:
            seg_rend.close()


if __name__ == "__main__":
    unittest.main()
