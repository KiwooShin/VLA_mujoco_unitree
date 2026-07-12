"""Unit tests for code.sim.arena_cameras (pure camera-frame math, no EGL/GPU)."""

import math
import unittest

import numpy as np
import mujoco

from code.sim.arena_cameras import _set_ego_cam, backproject_pixel, get_ego_intrinsics


class TestGetEgoIntrinsics(unittest.TestCase):
    def test_default_matches_ego_constants(self) -> None:
        intr = get_ego_intrinsics()
        self.assertEqual(intr["width"], 320)
        self.assertEqual(intr["height"], 240)
        self.assertEqual(intr["fovy_deg"], 90.0)

    def test_principal_point_is_pixel_center(self) -> None:
        intr = get_ego_intrinsics(w=320, h=240, fovy_deg=90.0)
        self.assertAlmostEqual(intr["cx"], 320 / 2.0 - 0.5)
        self.assertAlmostEqual(intr["cy"], 240 / 2.0 - 0.5)

    def test_fy_matches_pinhole_formula(self) -> None:
        w, h, fovy = 480, 360, 45.0
        intr = get_ego_intrinsics(w, h, fovy)
        expected_fy = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        self.assertAlmostEqual(intr["fy"], expected_fy)

    def test_square_pixels_when_w_equals_h(self) -> None:
        """With w==h, fovx==fovy so fx==fy exactly."""
        intr = get_ego_intrinsics(w=240, h=240, fovy_deg=90.0)
        self.assertAlmostEqual(intr["fx"], intr["fy"])

    def test_wider_aspect_gives_larger_fx_than_fy(self) -> None:
        intr = get_ego_intrinsics(w=480, h=240, fovy_deg=90.0)
        self.assertGreater(intr["fx"], intr["fy"])

    def test_keys_present(self) -> None:
        intr = get_ego_intrinsics()
        for key in ("fx", "fy", "cx", "cy", "width", "height", "fovy_deg"):
            self.assertIn(key, intr)


class TestBackprojectPixel(unittest.TestCase):
    def _intr(self) -> dict:
        return get_ego_intrinsics(w=320, h=240, fovy_deg=90.0)

    def test_center_pixel_lies_on_optical_axis(self) -> None:
        intr = self._intr()
        pt = backproject_pixel(intr["cx"], intr["cy"], 2.5, intr)
        np.testing.assert_allclose(pt, [0.0, 0.0, 2.5], atol=1e-5)

    def test_depth_zero_collapses_to_origin_ray(self) -> None:
        intr = self._intr()
        pt = backproject_pixel(10.0, 10.0, 0.0, intr)
        np.testing.assert_allclose(pt, [0.0, 0.0, 0.0], atol=1e-6)

    def test_returns_float32_array_shape_3(self) -> None:
        intr = self._intr()
        pt = backproject_pixel(5.0, 5.0, 1.0, intr)
        self.assertEqual(pt.shape, (3,))
        self.assertEqual(pt.dtype, np.float32)

    def test_project_backproject_roundtrip(self) -> None:
        """Forward pinhole projection (x,y,z)->(u,v) then backproject_pixel must
        return the original camera-frame point (up to float32 precision)."""
        intr = self._intr()
        x_cam, y_cam, z_cam = 0.4, -0.2, 3.0
        u = intr["fx"] * x_cam / z_cam + intr["cx"]
        v = intr["fy"] * y_cam / z_cam + intr["cy"]
        pt = backproject_pixel(u, v, z_cam, intr)
        np.testing.assert_allclose(pt, [x_cam, y_cam, z_cam], atol=1e-4)

    def test_right_and_down_signs(self) -> None:
        """u > cx should give positive x (right); v > cy should give positive y (down)."""
        intr = self._intr()
        pt = backproject_pixel(intr["cx"] + 50, intr["cy"] + 30, 2.0, intr)
        self.assertGreater(pt[0], 0.0)
        self.assertGreater(pt[1], 0.0)


class TestSetEgoCam(unittest.TestCase):
    def _cam(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return cam

    def test_distance_is_always_one(self) -> None:
        """P0 fix: distance=1.0 decouples eye position from pitch (see arena_cameras.py)."""
        cam = self._cam()
        qpos = np.array([1.0, 2.0, 0.79, 1, 0, 0, 0], dtype=np.float64)
        _set_ego_cam(cam, qpos, yaw=0.3, pitch_deg=26.0)
        self.assertEqual(cam.distance, 1.0)

    def test_azimuth_matches_yaw_in_degrees(self) -> None:
        cam = self._cam()
        qpos = np.zeros(7)
        yaw = math.radians(37.0)
        _set_ego_cam(cam, qpos, yaw=yaw, pitch_deg=32.0)
        self.assertAlmostEqual(cam.azimuth, 37.0, places=4)

    def test_elevation_is_negative_pitch(self) -> None:
        cam = self._cam()
        qpos = np.zeros(7)
        _set_ego_cam(cam, qpos, yaw=0.0, pitch_deg=26.0)
        self.assertAlmostEqual(cam.elevation, -26.0)

    def test_lookat_offset_from_origin_at_zero_yaw_zero_pitch(self) -> None:
        """At yaw=0, pitch=0: forward dir is exactly +X, so lookat.x = origin.x + 1."""
        cam = self._cam()
        qpos = np.array([0.0, 0.0, 0.79, 1, 0, 0, 0])
        _set_ego_cam(cam, qpos, yaw=0.0, pitch_deg=0.0)
        cam_head_z = 0.55  # CAM_HEAD_Z
        cam_fwd = 0.10      # CAM_FWD
        expected_origin_x = 0.0 + cam_fwd
        expected_origin_z = 0.79 + cam_head_z
        np.testing.assert_allclose(list(cam.lookat), [expected_origin_x + 1.0, 0.0, expected_origin_z], atol=1e-6)

    def test_default_pitch_is_cam_pitch(self) -> None:
        """Default pitch_deg parameter should equal CAM_PITCH (32 deg)."""
        cam_default = self._cam()
        cam_explicit = self._cam()
        qpos = np.array([0.5, -0.3, 0.79, 1, 0, 0, 0])
        _set_ego_cam(cam_default, qpos, yaw=0.4)
        _set_ego_cam(cam_explicit, qpos, yaw=0.4, pitch_deg=32.0)
        self.assertEqual(cam_default.elevation, cam_explicit.elevation)
        np.testing.assert_allclose(list(cam_default.lookat), list(cam_explicit.lookat))


if __name__ == "__main__":
    unittest.main()
