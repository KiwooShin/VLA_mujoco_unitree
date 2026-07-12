"""Unit tests for code.apps.fancy.overlays_projection: world->BEV-pixel
projection geometry + the small dashed-line/color-lerp drawing helpers.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from code.apps.fancy.overlays_projection import (
    TRAIL_COOL_BGR, TRAIL_WARM_BGR, _lerp_color_bgr, world_to_bev_pixel,
)


def _make_cam(azimuth: float, elevation: float, distance: float,
              lookat=(0.0, 0.0, 0.0)) -> SimpleNamespace:
    """Minimal stand-in for mujoco.MjvCamera exposing the 4 fields
    world_to_bev_pixel actually reads."""
    return SimpleNamespace(azimuth=azimuth, elevation=elevation,
                            distance=distance, lookat=list(lookat))


class WorldToBevPixelTest(unittest.TestCase):
    def test_lookat_point_projects_to_image_center(self) -> None:
        cam = _make_cam(azimuth=90.0, elevation=-45.0, distance=5.0, lookat=(1.0, 2.0, 0.0))
        pix = world_to_bev_pixel(np.array([1.0, 2.0, 0.0]), cam, None, None, w=640, h=480)
        self.assertEqual(pix.shape, (1, 2))
        u, v = pix[0]
        self.assertAlmostEqual(u, 640 / 2.0 - 0.5, places=3)
        self.assertAlmostEqual(v, 480 / 2.0 - 0.5, places=3)

    def test_single_point_1d_input_is_promoted_to_2d(self) -> None:
        cam = _make_cam(azimuth=0.0, elevation=-30.0, distance=3.0)
        pix = world_to_bev_pixel(np.array([0.5, 0.5, 0.0]), cam, None, None)
        self.assertEqual(pix.shape, (1, 2))

    def test_multiple_points_preserve_order_and_count(self) -> None:
        cam = _make_cam(azimuth=225.0, elevation=-43.5, distance=17.0, lookat=(0.0, 0.0, 0.3))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pix = world_to_bev_pixel(pts, cam, None, None, w=640, h=480)
        self.assertEqual(pix.shape, (3, 2))

    def test_points_on_either_side_of_boresight_land_on_opposite_sides(self) -> None:
        # Camera looking along +X (azimuth=0), lookat at the world origin.
        # Two points symmetric about the boresight line (+/-Y) must project
        # to pixel columns on opposite sides of the image's vertical center.
        cam = _make_cam(azimuth=0.0, elevation=-10.0, distance=5.0)
        pts = np.array([[3.0, 0.5, 0.0], [3.0, -0.5, 0.0]])
        pix = world_to_bev_pixel(pts, cam, None, None, w=640, h=480)
        cx = 640 / 2.0 - 0.5
        self.assertLess((pix[0, 0] - cx) * (pix[1, 0] - cx), 0.0)

    def test_custom_resolution_changes_principal_point(self) -> None:
        cam = _make_cam(azimuth=90.0, elevation=-45.0, distance=5.0, lookat=(1.0, 2.0, 0.0))
        pix = world_to_bev_pixel(np.array([1.0, 2.0, 0.0]), cam, None, None, w=320, h=240)
        u, v = pix[0]
        self.assertAlmostEqual(u, 320 / 2.0 - 0.5, places=3)
        self.assertAlmostEqual(v, 240 / 2.0 - 0.5, places=3)

    def test_behind_camera_point_is_clamped_not_inf(self) -> None:
        # A point coincident with the camera position (z_cam ~ 0) must not
        # produce inf/nan -- the z_cam_safe clamp guards against div-by-zero.
        cam = _make_cam(azimuth=0.0, elevation=0.0, distance=0.0)
        pix = world_to_bev_pixel(np.array([0.0, 0.0, 0.0]), cam, None, None)
        self.assertTrue(np.all(np.isfinite(pix)))


class LerpColorBgrTest(unittest.TestCase):
    def test_t0_returns_cool(self) -> None:
        self.assertEqual(_lerp_color_bgr((0, 0, 0), (100, 100, 100), 0.0), (0, 0, 0))

    def test_t1_returns_warm(self) -> None:
        self.assertEqual(_lerp_color_bgr((0, 0, 0), (100, 100, 100), 1.0), (100, 100, 100))

    def test_midpoint_averages(self) -> None:
        self.assertEqual(_lerp_color_bgr((0, 0, 0), (100, 200, 50), 0.5), (50, 100, 25))

    def test_clamped_below_zero(self) -> None:
        self.assertEqual(_lerp_color_bgr((10, 10, 10), (20, 20, 20), -5.0), (10, 10, 10))

    def test_clamped_above_one(self) -> None:
        self.assertEqual(_lerp_color_bgr((10, 10, 10), (20, 20, 20), 5.0), (20, 20, 20))

    def test_trail_gradient_endpoints_are_distinct(self) -> None:
        self.assertNotEqual(TRAIL_COOL_BGR, TRAIL_WARM_BGR)


class DashedLineTest(unittest.TestCase):
    def test_zero_length_segment_is_noop(self) -> None:
        import cv2
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        from code.apps.fancy.overlays_projection import _dashed_line
        _dashed_line(img, (5, 5), (5, 5), (255, 255, 255))
        self.assertTrue(np.all(img == 0))

    def test_draws_some_nonzero_pixels_for_a_real_segment(self) -> None:
        from code.apps.fancy.overlays_projection import _dashed_line
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        _dashed_line(img, (2, 25), (47, 25), (255, 255, 255), thickness=1)
        self.assertGreater(int(img.sum()), 0)


if __name__ == "__main__":
    unittest.main()
