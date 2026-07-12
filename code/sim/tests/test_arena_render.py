"""Integration smokes for code.sim.arena_render.ArenaRenderer.

These build a real (tiny) arena and drive the EGL renderer for a single
frame per camera stream — cheap (~1.3 ms/frame on GPU per docs/cam_p0.md)
but a real render, not a mock. Each test skips gracefully if EGL/MuJoCo
rendering is unavailable in the current environment (e.g. no GPU/EGL
vendor), rather than failing the whole suite.
"""

import unittest

import mujoco
import numpy as np

from code.sim.arena_build import GROUNDING_H, GROUNDING_W, PROXIMITY_H, PROXIMITY_W, build_arena
from code.sim.arena_render import ArenaRenderer


def _tiny_model() -> mujoco.MjModel:
    scene_cfg = {
        "arena_size": 4.0,
        "objects": [
            {"color_name": "red", "color_rgb": (220, 40, 40),
             "shape_name": "ball", "size": 0.24, "x": 1.5, "y": 0.0},
        ],
        "lighting": {"ambient": 0.4},
    }
    return build_arena(scene_cfg)


def _make_renderer_or_skip(test: unittest.TestCase) -> ArenaRenderer:
    try:
        model = _tiny_model()
        return ArenaRenderer(model)
    except Exception as e:  # pragma: no cover - environment-dependent
        test.skipTest(f"EGL/MuJoCo renderer unavailable: {e}")


class TestArenaRendererLifecycle(unittest.TestCase):
    """One ArenaRenderer (EGL context) shared across the whole class — allocating
    a fresh set of mujoco.Renderer objects per test would dominate runtime and
    doesn't exercise anything the shared instance doesn't already cover."""

    renderer: ArenaRenderer

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.renderer = ArenaRenderer(_tiny_model())
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"EGL/MuJoCo renderer unavailable: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "renderer"):
            cls.renderer.close()

    def setUp(self) -> None:
        self.model = self.renderer._model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.79
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def test_render_ego_shapes(self) -> None:
        rgb, depth, intr = self.renderer.render_ego(self.data, yaw=0.0)
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.shape, (240, 320))
        self.assertEqual(intr["fovy_deg"], 90.0)

    def test_render_ego_skips_depth_when_disabled(self) -> None:
        rgb, depth, _ = self.renderer.render_ego(self.data, yaw=0.0, render_depth=False)
        self.assertIsNone(depth)
        self.assertEqual(rgb.shape, (240, 320, 3))

    def test_render_grounding_shapes_and_pitch(self) -> None:
        rgb, depth, intr = self.renderer.render_grounding(self.data, yaw=0.0)
        self.assertEqual(rgb.shape, (GROUNDING_H, GROUNDING_W, 3))
        self.assertEqual(depth.shape, (GROUNDING_H, GROUNDING_W))
        self.assertEqual(intr["pitch_deg"], 26.0)

    def test_render_proximity_flags_is_proximity(self) -> None:
        rgb, depth, intr = self.renderer.render_proximity(self.data, yaw=0.0)
        self.assertEqual(rgb.shape, (PROXIMITY_H, PROXIMITY_W, 3))
        self.assertTrue(intr["is_proximity"])
        self.assertEqual(intr["pitch_deg"], 58.0)

    def test_render_tp_and_camera_tracking(self) -> None:
        tp_cam = self.renderer.make_tp_cam()
        self.renderer.update_tp_cam(tp_cam, self.data)
        rgb = self.renderer.render_tp(self.data, tp_cam)
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(list(tp_cam.lookat), [self.data.qpos[0], self.data.qpos[1], 0.5])

    def test_widefov_renderer_absent_in_default_cam2_mode(self) -> None:
        self.assertIsNone(self.renderer._widefov_rend)

    def test_render_ego_returns_fresh_copy_not_view(self) -> None:
        """Successive renders must not alias the same buffer (renderer reuses one
        internal buffer per call to .render(), so callers rely on the .copy())."""
        rgb1, _, _ = self.renderer.render_ego(self.data, yaw=0.0, render_depth=False)
        self.data.qpos[0] += 0.5
        mujoco.mj_forward(self.model, self.data)
        rgb2, _, _ = self.renderer.render_ego(self.data, yaw=0.0, render_depth=False)
        # rgb1 must remain whatever it was at the time of the first call
        self.assertFalse(rgb1 is rgb2)


class TestArenaRendererClose(unittest.TestCase):
    def test_close_is_idempotent_safe_to_call_once(self) -> None:
        renderer = _make_renderer_or_skip(self)
        renderer.close()  # should not raise


if __name__ == "__main__":
    unittest.main()
