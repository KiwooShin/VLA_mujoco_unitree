"""Unit tests for code/perception/geometry.py: intrinsics + camera-frame ->
egocentric transform (pure math, no rendering)."""
from __future__ import annotations

import math
import unittest

from code.perception.geometry import (CAM_PITCH_RAD, CAM_ROBOT_FORWARD_OFFSET_M,
                                       EGO_FOVY_RENDERED, cam_to_egocentric,
                                       get_ego_intrinsics_rendered)


class TestGetEgoIntrinsicsRendered(unittest.TestCase):

    def test_default_size_matches_arena_ego(self):
        intr = get_ego_intrinsics_rendered()
        self.assertEqual(intr["width"], 320)
        self.assertEqual(intr["height"], 240)
        self.assertEqual(intr["fovy_deg"], EGO_FOVY_RENDERED)

    def test_principal_point_is_center_minus_half(self):
        intr = get_ego_intrinsics_rendered(100, 50)
        self.assertAlmostEqual(intr["cx"], 100 / 2.0 - 0.5)
        self.assertAlmostEqual(intr["cy"], 50 / 2.0 - 0.5)

    def test_focal_lengths_positive_and_scale_with_size(self):
        small = get_ego_intrinsics_rendered(100, 75)
        large = get_ego_intrinsics_rendered(200, 150)
        self.assertGreater(small["fx"], 0)
        self.assertGreater(small["fy"], 0)
        # Doubling resolution at fixed FOV doubles focal length in pixels.
        self.assertAlmostEqual(large["fx"], 2 * small["fx"], places=6)
        self.assertAlmostEqual(large["fy"], 2 * small["fy"], places=6)

    def test_fy_matches_closed_form(self):
        w, h = 192, 144
        intr = get_ego_intrinsics_rendered(w, h)
        fovy_rad = math.radians(EGO_FOVY_RENDERED)
        expected_fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
        self.assertAlmostEqual(intr["fy"], expected_fy, places=6)


class TestCamToEgocentric(unittest.TestCase):

    def test_straight_ahead_zero_pitch_zero_bearing(self):
        # Directly in front, no pitch: x_cam=0, y_cam=0, z_cam=+5 -> straight ahead.
        dist, yaw = cam_to_egocentric(0.0, 0.0, 5.0, pitch_deg=0.0)
        self.assertAlmostEqual(yaw, 0.0, places=9)
        self.assertAlmostEqual(dist, 5.0 + CAM_ROBOT_FORWARD_OFFSET_M, places=6)

    def test_left_of_image_is_positive_yaw(self):
        # Negative x_cam (image-left) should map to a POSITIVE yaw_err (turn left).
        _, yaw_left = cam_to_egocentric(-1.0, 0.0, 5.0, pitch_deg=0.0)
        _, yaw_right = cam_to_egocentric(1.0, 0.0, 5.0, pitch_deg=0.0)
        self.assertGreater(yaw_left, 0.0)
        self.assertLess(yaw_right, 0.0)
        # Symmetric magnitude.
        self.assertAlmostEqual(yaw_left, -yaw_right, places=9)

    def test_default_pitch_matches_cam_pitch_rad(self):
        # cam_to_egocentric's default pitch_deg argument is derived from CAM_PITCH_RAD.
        dist_a, yaw_a = cam_to_egocentric(0.3, 0.1, 4.0)
        dist_b, yaw_b = cam_to_egocentric(0.3, 0.1, 4.0, pitch_deg=math.degrees(CAM_PITCH_RAD))
        self.assertAlmostEqual(dist_a, dist_b, places=9)
        self.assertAlmostEqual(yaw_a, yaw_b, places=9)

    def test_uncorrected_vs_corrected_unpitch_differ_at_steep_pitch(self):
        # At a steep pitch the pre-existing (buggy) sign and the CAM-2 corrected
        # sign should diverge measurably -- this is the whole point of the flag.
        x, y, z = 0.2, 0.5, 3.0
        dist_old, _ = cam_to_egocentric(x, y, z, pitch_deg=58.0, use_corrected_unpitch=False)
        dist_new, _ = cam_to_egocentric(x, y, z, pitch_deg=58.0, use_corrected_unpitch=True)
        self.assertNotAlmostEqual(dist_old, dist_new, places=3)

    def test_corrected_unpitch_distance_monotonic_in_true_range(self):
        # docs/cam_p1.md: with the corrected sign, reported distance must increase
        # monotonically as the true forward distance (z_cam) increases, at a steep
        # pitch where the uncorrected formula was documented to misbehave.
        dists = []
        for z_cam in (1.0, 2.0, 3.0, 4.0, 5.0):
            d, _ = cam_to_egocentric(0.0, 0.3, z_cam, pitch_deg=58.0, use_corrected_unpitch=True)
            dists.append(d)
        self.assertEqual(dists, sorted(dists))
        # Must be strictly increasing (no plateaus/inversions).
        for a, b in zip(dists, dists[1:]):
            self.assertLess(a, b)

    def test_forward_offset_is_added_to_forward_distance(self):
        dist_a, _ = cam_to_egocentric(0.0, 0.0, 1.0, pitch_deg=0.0)
        # z_robot ends up 1.0 + CAM_ROBOT_FORWARD_OFFSET_M with x_robot=0, so dist == that sum.
        self.assertAlmostEqual(dist_a, 1.0 + CAM_ROBOT_FORWARD_OFFSET_M, places=9)

    def test_dist_is_nonnegative_hypot(self):
        for x, y, z in [(-3, 2, -5), (0, 0, 0), (10, -10, 10)]:
            dist, _ = cam_to_egocentric(x, y, z, pitch_deg=20.0)
            self.assertGreaterEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
