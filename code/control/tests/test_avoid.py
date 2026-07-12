"""Unit tests for code.control.avoid (RF-1: core + geometry submodules).

Ports the synthetic-depth-frame assertions from the old code/avoid.py
`if __name__ == "__main__":` self-test (see code/control/avoid/_selftest.py,
still runnable via `python code/avoid.py`) into proper unittest cases, plus
additional edge/determinism coverage for the carve-outs and hysteresis.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.control.avoid import core as C
from code.control.avoid import geometry as G
from code.control.steer import MAX_WZ
from code.grounding import CAM_ROBOT_FORWARD_OFFSET_M
from code.arena import GROUNDING_PITCH

W, H = 480, 360
CAM_H = 1.34  # RESET_HEIGHT(0.79) + CAM_HEAD_Z(0.55), approx walking height


def _get_intr() -> dict:
    from code.grounding import get_ego_intrinsics_rendered
    intr = get_ego_intrinsics_rendered(W, H)
    intr['pitch_deg'] = GROUNDING_PITCH
    return intr


def _inverse_uncorrected(x_robot, y_vert, z_robot_raw, intr):
    pitch_rad = math.radians(intr['pitch_deg'])
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    A = np.array([[sp, cp], [cp, sp]], dtype=np.float64)
    b = np.array([z_robot_raw, y_vert], dtype=np.float64)
    y_cam, z_cam = np.linalg.solve(A, b)
    u = intr['cx'] + x_robot * intr['fx'] / z_cam
    v = intr['cy'] + y_cam * intr['fy'] / z_cam
    return u, v, z_cam


def _blank_floor_frame(intr, near_m=0.4, far_m=6.0):
    depth = np.full((H, W), far_m, dtype=np.float32)
    pitch_rad = math.radians(intr['pitch_deg'])
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    fy, cy = intr['fy'], intr['cy']
    for v in range(H):
        denom = sp + cp * (v - cy) / fy
        if abs(denom) < 1e-6:
            continue
        d = CAM_H / denom
        if not math.isfinite(d) or d <= 0:
            continue
        depth[v, :] = float(np.clip(d, near_m, far_m))
    return depth


def _wall_frame(intr, bearing_deg_lo, bearing_deg_hi, dist_m,
                 world_height_m=1.1, background_m=6.0):
    depth = np.full((H, W), background_m, dtype=np.float32)
    y_vert = CAM_H - world_height_m
    for bearing_deg in np.linspace(bearing_deg_lo, bearing_deg_hi, 400):
        bearing_rad = math.radians(bearing_deg)
        x_robot = -dist_m * math.sin(bearing_rad)
        z_robot = dist_m * math.cos(bearing_rad)
        z_robot_raw = z_robot - CAM_ROBOT_FORWARD_OFFSET_M
        u, v, z_cam = _inverse_uncorrected(x_robot, y_vert, z_robot_raw, intr)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < W and z_cam > 0:
            depth[max(0, vi - 15):min(H, vi + 15), ui] = max(z_cam, 0.05)
    return depth


class AvoidTestBase(unittest.TestCase):
    """Shared fixtures for all avoid.py-derived synthetic-frame tests."""

    @classmethod
    def setUpClass(cls):
        cls.intr = _get_intr()
        cls.floor_depth = _blank_floor_frame(cls.intr)
        cls.clear_depth = np.full((H, W), 6.0, dtype=np.float32)
        cls.wall_left = _wall_frame(cls.intr, 5.0, 22.0, dist_m=0.5)
        cls.wall_right = _wall_frame(cls.intr, -22.0, -5.0, dist_m=0.5)
        cls.wall_center = _wall_frame(cls.intr, -20.0, 20.0, dist_m=0.5)
        cls.target_blob = _wall_frame(cls.intr, -6.0, 6.0, dist_m=0.5)


class TestComputeObstacleBiasFloorAndClear(AvoidTestBase):
    def test_floor_only_zero_bias(self):
        bias, info = C.compute_obstacle_bias(self.floor_depth, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertEqual(bias, 0.0)
        self.assertEqual(info['n_obstacle_px'], 0)

    def test_clear_frame_zero_bias(self):
        bias, info = C.compute_obstacle_bias(self.clear_depth, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertEqual(bias, 0.0)


class TestComputeObstacleBiasDirection(AvoidTestBase):
    def test_wall_left_steers_right_negative_wz(self):
        bias, info = C.compute_obstacle_bias(self.wall_left, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertLess(bias, -0.01)
        self.assertGreater(info['left'], info['right'])

    def test_wall_right_steers_left_positive_wz(self):
        bias, info = C.compute_obstacle_bias(self.wall_right, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertGreater(bias, 0.01)
        self.assertGreater(info['right'], info['left'])

    def test_wall_center_decisive_tie_break_within_cap(self):
        bias, info = C.compute_obstacle_bias(self.wall_center, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertGreater(abs(bias), 0.01)
        self.assertLessEqual(abs(bias), C.AVOID_MAX_WZ_BIAS + 1e-6)


class TestTargetExemption(AvoidTestBase):
    def test_exempted_when_goal_close(self):
        bias, info = C.compute_obstacle_bias(self.target_blob, self.intr, CAM_H,
                                              goal_dist_m=1.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertEqual(bias, 0.0)
        self.assertEqual(info['n_obstacle_px'], 0)

    def test_not_exempted_when_goal_far(self):
        bias, info = C.compute_obstacle_bias(self.target_blob, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.0)
        self.assertGreater(abs(bias), 0.01)

    def test_exemption_window_tracks_goal_bearing(self):
        # Move the blob to +15 deg and the goal bearing to match -> should exempt again.
        blob = _wall_frame(self.intr, 9.0, 21.0, dist_m=0.5)
        bias, info = C.compute_obstacle_bias(blob, self.intr, CAM_H,
                                              goal_dist_m=1.0,
                                              goal_bearing_rad=math.radians(15.0),
                                              prev_bias_wz=0.0)
        self.assertEqual(info['n_obstacle_px'], 0)


class TestCarveOuts(AvoidTestBase):
    def test_carved_out_hard_zeros(self):
        bias, info = C.compute_obstacle_bias(self.wall_left, self.intr, CAM_H,
                                              goal_dist_m=0.8, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.25, carved_out=True)
        self.assertEqual(bias, 0.0)
        self.assertTrue(info['carved_out'])

    def test_carved_out_ignores_prior_bias_entirely(self):
        bias, info = C.compute_obstacle_bias(self.clear_depth, self.intr, CAM_H,
                                              goal_dist_m=5.0, goal_bearing_rad=0.0,
                                              prev_bias_wz=0.29, carved_out=True)
        self.assertEqual(bias, 0.0)

    def test_is_maneuver_scene_true(self):
        self.assertTrue(C.is_maneuver_scene({'difficulty': 'maneuver'}))
        self.assertTrue(C.is_maneuver_scene({'difficulty': 'MANEUVER'}))

    def test_is_maneuver_scene_false(self):
        self.assertFalse(C.is_maneuver_scene({'difficulty': 'demo'}))
        self.assertFalse(C.is_maneuver_scene({}))


class TestHysteresisAndDecay(AvoidTestBase):
    def test_gradual_decay_not_instant_zero(self):
        b = 0.28
        b, _ = C.compute_obstacle_bias(self.clear_depth, self.intr, CAM_H,
                                        goal_dist_m=5.0, goal_bearing_rad=0.0,
                                        prev_bias_wz=b)
        self.assertGreater(b, 0.0)
        self.assertLess(b, 0.28)

    def test_reaches_zero_within_five_cycles(self):
        b = 0.28
        for _ in range(6):
            b, _ = C.compute_obstacle_bias(self.clear_depth, self.intr, CAM_H,
                                            goal_dist_m=5.0, goal_bearing_rad=0.0,
                                            prev_bias_wz=b)
        self.assertEqual(b, 0.0)

    def test_decay_bias_function_matches_schedule(self):
        b0 = 0.2
        b1 = C.decay_bias(b0)
        self.assertAlmostEqual(b1, b0 * C.AVOID_DECAY_FACTOR, places=9)

    def test_decay_bias_snaps_to_zero_below_deadband(self):
        tiny = C.AVOID_DEADBAND * C.AVOID_DECAY_FACTOR * 0.5  # decays to below deadband
        self.assertEqual(C.decay_bias(tiny), 0.0)

    def test_decay_bias_clips_to_max(self):
        self.assertLessEqual(abs(C.decay_bias(999.0)), C.AVOID_MAX_WZ_BIAS + 1e-9)

    def test_ema_blend_toward_fresh_raw_bias(self):
        # A persistent obstacle (same wall each cycle) should converge, not oscillate
        # wildly, and stay within the cap.
        b = 0.0
        trace = []
        for _ in range(5):
            b, _ = C.compute_obstacle_bias(self.wall_left, self.intr, CAM_H,
                                            goal_dist_m=5.0, goal_bearing_rad=0.0,
                                            prev_bias_wz=b)
            trace.append(b)
        self.assertTrue(all(abs(x) <= C.AVOID_MAX_WZ_BIAS + 1e-9 for x in trace))
        # Monotonically approaching a negative steady value from 0.
        self.assertTrue(all(trace[i] <= trace[i - 1] + 1e-9 for i in range(1, len(trace))))


class TestBiasedVelCmd(unittest.TestCase):
    def test_wz_reflects_bias_vy_zero(self):
        vel = C.biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=0.3, stop_r=0.6)
        self.assertAlmostEqual(float(vel[2]), 0.3, places=5)
        self.assertEqual(vel[1], 0.0)

    def test_clipped_to_steer_max_wz(self):
        vel = C.biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=5.0, stop_r=0.6)
        self.assertLessEqual(abs(vel[2]), MAX_WZ + 1e-6)

    def test_within_stop_r_zeros_regardless_of_bias(self):
        vel = C.biased_vel_cmd(goal_dist=0.3, cos_th=1.0, sin_th=0.0, bias_wz=5.0, stop_r=0.6)
        self.assertTrue(np.allclose(vel, 0.0))

    def test_output_dtype_and_shape(self):
        vel = C.biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=0.0, stop_r=0.6)
        self.assertEqual(vel.shape, (3,))
        self.assertEqual(vel.dtype, np.float32)

    def test_negative_bias_can_flip_wz_sign(self):
        vel = C.biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=-0.05, stop_r=0.6)
        self.assertLess(float(vel[2]), 0.0)


class TestGeometryBackproject(unittest.TestCase):
    """code.control.avoid.geometry.backproject_frame."""

    @classmethod
    def setUpClass(cls):
        cls.intr = _get_intr()

    def test_output_shapes_and_dtypes(self):
        depth = np.full((H, W), 3.0, dtype=np.float32)
        dist, bearing, y_vert = G.backproject_frame(depth, self.intr)
        for arr in (dist, bearing, y_vert):
            self.assertEqual(arr.shape, (H, W))
            self.assertEqual(arr.dtype, np.float32)

    def test_center_pixel_bearing_near_zero(self):
        depth = np.full((H, W), 3.0, dtype=np.float32)
        dist, bearing, _ = G.backproject_frame(depth, self.intr)
        cy, cx = int(self.intr['cy']), int(self.intr['cx'])
        self.assertAlmostEqual(float(bearing[cy, cx]), 0.0, places=2)

    def test_private_alias_matches_public(self):
        self.assertIs(G._backproject_frame, G.backproject_frame)

    def test_floor_pixel_height_near_zero(self):
        floor_depth = _blank_floor_frame(self.intr)
        dist, bearing, y_vert = G.backproject_frame(floor_depth, self.intr)
        height_above_ground = CAM_H - y_vert
        # Most rows in the floor synth frame should read ~0 height above ground.
        mid = H // 2
        self.assertLess(abs(float(height_above_ground[mid, W // 2])), 0.05)


class TestEnvFlagAndConstants(unittest.TestCase):
    def test_env_flag_default_false(self):
        self.assertFalse(C._env_flag("SOME_NONEXISTENT_AVOID_TEST_FLAG_XYZ"))

    def test_env_flag_true_string(self):
        import os
        os.environ["SOME_TEST_FLAG_AVOID_XYZ"] = "1"
        try:
            self.assertTrue(C._env_flag("SOME_TEST_FLAG_AVOID_XYZ"))
        finally:
            del os.environ["SOME_TEST_FLAG_AVOID_XYZ"]

    def test_min_goal_dist_is_1_6(self):
        self.assertEqual(C.AVOID_MIN_GOAL_DIST_M, 1.6)

    def test_corridor_half_deg_is_25(self):
        self.assertEqual(C.AVOID_CORRIDOR_HALF_DEG, 25.0)

    def test_stale_max_missed_cycles_is_2(self):
        self.assertEqual(C.AVOID_STALE_MAX_MISSED_CYCLES, 2)


if __name__ == "__main__":
    unittest.main()
