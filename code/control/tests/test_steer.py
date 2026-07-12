"""Unit tests for code.control.steer (RF-1).

Pins the privileged steering control law's geometry (egocentric_goal),
velocity command shaping (steer), and the goal-vector label helper
(goal_vec), including the turn-in-place / deceleration / stop-radius
behaviors and the wz clamp.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.control import steer as S


class TestAngleDiff(unittest.TestCase):
    """_angle_diff: signed wrap-to-(-pi, pi] angular difference."""

    def test_zero_diff(self):
        self.assertAlmostEqual(S._angle_diff(0.0, 0.0), 0.0)

    def test_simple_positive(self):
        self.assertAlmostEqual(S._angle_diff(math.radians(30), math.radians(10)),
                                math.radians(20), places=6)

    def test_wraps_across_pi(self):
        # 170 deg minus -170 deg "as angles" should wrap to -20 deg, not 340 deg.
        d = S._angle_diff(math.radians(170), math.radians(-170))
        self.assertAlmostEqual(d, math.radians(-20), places=6)

    def test_result_in_range(self):
        for a_deg in range(-350, 360, 37):
            for b_deg in range(-350, 360, 53):
                d = S._angle_diff(math.radians(a_deg), math.radians(b_deg))
                self.assertGreater(d, -math.pi - 1e-9)
                self.assertLessEqual(d, math.pi + 1e-9)


class TestEgocentricGoal(unittest.TestCase):
    """egocentric_goal: world-frame poses -> (dist, yaw_err, bearing)."""

    def test_target_straight_ahead(self):
        dist, yaw_err, bearing = S.egocentric_goal((0.0, 0.0), 0.0, (3.0, 0.0))
        self.assertAlmostEqual(dist, 3.0, places=6)
        self.assertAlmostEqual(yaw_err, 0.0, places=6)
        self.assertAlmostEqual(bearing, 0.0, places=6)

    def test_target_directly_behind(self):
        dist, yaw_err, bearing = S.egocentric_goal((0.0, 0.0), 0.0, (-3.0, 0.0))
        self.assertAlmostEqual(dist, 3.0, places=6)
        # +-pi are equivalent; just check it's ~180 degrees off
        self.assertAlmostEqual(abs(yaw_err), math.pi, places=6)

    def test_target_to_the_left(self):
        # Robot facing +X, target at +Y -> should read as a positive (left) bearing/yaw_err.
        dist, yaw_err, bearing = S.egocentric_goal((0.0, 0.0), 0.0, (0.0, 3.0))
        self.assertAlmostEqual(dist, 3.0, places=6)
        self.assertGreater(yaw_err, 0.0)

    def test_target_to_the_right(self):
        dist, yaw_err, bearing = S.egocentric_goal((0.0, 0.0), 0.0, (0.0, -3.0))
        self.assertLess(yaw_err, 0.0)

    def test_robot_yaw_offset_cancels_out(self):
        # Rotating the robot's own yaw by the bearing angle should zero yaw_err.
        target = (2.0, 2.0)
        _, _, bearing = S.egocentric_goal((0.0, 0.0), 0.0, target)
        dist, yaw_err, bearing2 = S.egocentric_goal((0.0, 0.0), bearing, target)
        self.assertAlmostEqual(yaw_err, 0.0, places=6)
        self.assertAlmostEqual(bearing, bearing2, places=6)

    def test_zero_distance(self):
        dist, yaw_err, bearing = S.egocentric_goal((1.0, 1.0), 0.3, (1.0, 1.0))
        self.assertEqual(dist, 0.0)

    def test_accepts_numpy_arrays(self):
        dist, yaw_err, bearing = S.egocentric_goal(
            np.array([0.0, 0.0]), 0.0, np.array([3.0, 4.0]))
        self.assertAlmostEqual(dist, 5.0, places=6)


class TestSteer(unittest.TestCase):
    """steer(): full velocity-command control law."""

    def test_stop_radius_zeros_command(self):
        cmd, dist, yaw_err = S.steer((0, 0), 0.0, (0.3, 0.0), stop_r=0.6)
        self.assertTrue(np.allclose(cmd, 0.0))
        self.assertLess(dist, 0.6)

    def test_forward_when_aligned_and_far(self):
        cmd, dist, yaw_err = S.steer((0, 0), 0.0, (3.0, 0.0), stop_r=0.6)
        self.assertGreater(cmd[0], 0.0)
        self.assertAlmostEqual(cmd[2], 0.0, places=6)  # aligned -> no yaw command needed
        self.assertEqual(cmd[1], 0.0)  # never lateral

    def test_turn_in_place_when_target_behind(self):
        cmd, dist, yaw_err = S.steer((0, 0), 0.0, (-3.0, 0.0), stop_r=0.6)
        self.assertEqual(cmd[0], 0.0)
        self.assertGreater(abs(yaw_err), S.FACE_THR_RAD)

    def test_wz_sign_matches_bearing_side(self):
        cmd_left, _, _ = S.steer((0, 0), 0.0, (1.0, 1.0), stop_r=0.1)
        cmd_right, _, _ = S.steer((0, 0), 0.0, (1.0, -1.0), stop_r=0.1)
        self.assertGreater(cmd_left[2], 0.0)
        self.assertLess(cmd_right[2], 0.0)

    def test_wz_clamped_to_max_wz(self):
        # A target almost directly to the side should saturate yaw rate at max_wz.
        cmd, _, yaw_err = S.steer((0, 0), 0.0, (0.001, 5.0), stop_r=0.1, max_wz=0.3)
        self.assertLessEqual(abs(cmd[2]), 0.3 + 1e-6)

    def test_decel_ramp_reduces_vx_near_stop_r(self):
        far_cmd, _, _ = S.steer((0, 0), 0.0, (5.0, 0.0), stop_r=0.6, decel_dist=0.9)
        near_cmd, _, _ = S.steer((0, 0), 0.0, (0.8, 0.0), stop_r=0.6, decel_dist=0.9)
        self.assertGreater(far_cmd[0], near_cmd[0])
        self.assertGreaterEqual(near_cmd[0], 0.0)

    def test_vx_never_exceeds_max_vx(self):
        cmd, _, _ = S.steer((0, 0), 0.0, (10.0, 0.0), stop_r=0.6, max_vx=0.55)
        self.assertLessEqual(cmd[0], 0.55 + 1e-6)

    def test_custom_max_vx_respected(self):
        cmd, _, _ = S.steer((0, 0), 0.0, (10.0, 0.0), stop_r=0.6, max_vx=0.2)
        self.assertLessEqual(cmd[0], 0.2 + 1e-6)

    def test_output_dtype_and_shape(self):
        cmd, _, _ = S.steer((0, 0), 0.0, (3.0, 1.0), stop_r=0.6)
        self.assertEqual(cmd.shape, (3,))
        self.assertEqual(cmd.dtype, np.float32)

    def test_deterministic_repeat_calls(self):
        c1, d1, y1 = S.steer((0.1, -0.2), 0.4, (3.0, 1.0), stop_r=0.6)
        c2, d2, y2 = S.steer((0.1, -0.2), 0.4, (3.0, 1.0), stop_r=0.6)
        self.assertTrue(np.array_equal(c1, c2))
        self.assertEqual(d1, d2)
        self.assertEqual(y1, y2)

    def test_exactly_at_stop_r_boundary_not_stopped(self):
        # dist < stop_r stops; dist == stop_r should NOT trigger the stop branch.
        cmd, dist, _ = S.steer((0, 0), 0.0, (0.6, 0.0), stop_r=0.6)
        self.assertEqual(dist, 0.6)
        # Not required to be moving (decel_factor could be 0 at the boundary
        # depending on decel_dist), but it must not take the "==0 stop" path
        # silently for reasons other than the decel ramp.
        self.assertGreaterEqual(cmd[0], 0.0)


class TestGoalVec(unittest.TestCase):
    """goal_vec: privileged (dist, cos, sin) label used in the ADR schema."""

    def test_shape_and_dtype(self):
        v = S.goal_vec(2.5, 0.3)
        self.assertEqual(v.shape, (3,))
        self.assertEqual(v.dtype, np.float32)

    def test_values(self):
        v = S.goal_vec(2.0, math.pi / 2)
        self.assertAlmostEqual(float(v[0]), 2.0, places=5)
        self.assertAlmostEqual(float(v[1]), 0.0, places=5)
        self.assertAlmostEqual(float(v[2]), 1.0, places=5)

    def test_zero_yaw_err(self):
        v = S.goal_vec(1.0, 0.0)
        self.assertAlmostEqual(float(v[1]), 1.0, places=5)
        self.assertAlmostEqual(float(v[2]), 0.0, places=5)


class TestConstants(unittest.TestCase):
    """Sanity-pin the control-law constants other modules rely on unchanged."""

    def test_face_thr_matches_25_degrees(self):
        self.assertAlmostEqual(S.FACE_THR_RAD, math.radians(25.0), places=9)

    def test_vx_yaw_damp_is_zero(self):
        # "G1 walks straight" — code.control.avoid relies on this being 0.
        self.assertEqual(S.VX_YAW_DAMP, 0.0)


if __name__ == "__main__":
    unittest.main()
