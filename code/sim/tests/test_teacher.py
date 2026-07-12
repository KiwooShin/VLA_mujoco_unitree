"""Unit tests for code.sim.teacher (WBCTeacher core: helpers + class).

Pure-math helpers (_grav_orient, _yaw_of) get exhaustive coverage. WBCTeacher
itself needs a real ONNX session + MuJoCo model, so its tests are a small
number of cheap integration smokes (a handful of 50 Hz steps), skipped
gracefully if the ONNX policy / MuJoCo assets aren't available in this
checkout.
"""

import math
import os
import unittest

import numpy as np

from code.sim.teacher import (
    CMD_SCALE,
    CONTROL_DECIMATION,
    DEFAULT_ANGLES,
    G1_XML,
    KDS,
    KPS,
    NUM_ACTIONS,
    RESET_HEIGHT,
    SIM_DT,
    WALK_ONNX,
    WBCTeacher,
    _grav_orient,
    _yaw_of,
)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class TestGravOrient(unittest.TestCase):
    def test_identity_quat_gives_minus_z(self) -> None:
        g = _grav_orient(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)

    def test_output_is_unit_length(self) -> None:
        for yaw in (0.0, 0.3, math.pi / 2, math.pi, -1.2):
            g = _grav_orient(_quat_from_yaw(yaw))
            self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, places=5)

    def test_yaw_only_rotation_leaves_gravity_unchanged(self) -> None:
        """Gravity [0,0,-1] is invariant under any rotation about world Z."""
        for yaw in (0.1, 1.0, 2.5, -0.7):
            g = _grav_orient(_quat_from_yaw(yaw))
            np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)

    def test_pitch_rotation_tilts_gravity_xy(self) -> None:
        """A pure pitch (rotation about Y) should introduce a nonzero x-component."""
        half = math.radians(30.0) / 2.0
        quat = np.array([math.cos(half), 0.0, math.sin(half), 0.0])  # rotate about Y
        g = _grav_orient(quat)
        self.assertGreater(abs(g[0]), 1e-3)

    def test_returns_float32(self) -> None:
        g = _grav_orient(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(g.dtype, np.float32)


class TestYawOf(unittest.TestCase):
    def test_zero_yaw(self) -> None:
        self.assertAlmostEqual(_yaw_of(_quat_from_yaw(0.0)), 0.0, places=6)

    def test_quarter_turn(self) -> None:
        self.assertAlmostEqual(_yaw_of(_quat_from_yaw(math.pi / 2)), math.pi / 2, places=6)

    def test_half_turn(self) -> None:
        # atan2 wraps at +/-pi; a half-turn quat can yield either sign.
        result = _yaw_of(_quat_from_yaw(math.pi))
        self.assertAlmostEqual(abs(result), math.pi, places=5)

    def test_negative_yaw(self) -> None:
        self.assertAlmostEqual(_yaw_of(_quat_from_yaw(-1.0)), -1.0, places=6)

    def test_roundtrip_over_range(self) -> None:
        for yaw in np.linspace(-3.0, 3.0, 25):
            recovered = _yaw_of(_quat_from_yaw(float(yaw)))
            self.assertAlmostEqual(recovered, float(yaw), places=5)


class TestConstants(unittest.TestCase):
    def test_default_angles_length_matches_num_actions(self) -> None:
        self.assertEqual(len(DEFAULT_ANGLES), NUM_ACTIONS)

    def test_kps_kds_length_matches_num_actions(self) -> None:
        self.assertEqual(len(KPS), NUM_ACTIONS)
        self.assertEqual(len(KDS), NUM_ACTIONS)

    def test_gains_are_positive(self) -> None:
        self.assertTrue(np.all(KPS > 0))
        self.assertTrue(np.all(KDS > 0))

    def test_control_dt_matches_sim_dt_times_decimation(self) -> None:
        from code.sim.teacher import CONTROL_DT
        self.assertAlmostEqual(CONTROL_DT, SIM_DT * CONTROL_DECIMATION)
        self.assertAlmostEqual(CONTROL_DT, 0.02)

    def test_cmd_scale_length_3(self) -> None:
        self.assertEqual(len(CMD_SCALE), 3)

    def test_reset_height_is_standing_height(self) -> None:
        self.assertGreater(RESET_HEIGHT, 0.5)
        self.assertLess(RESET_HEIGHT, 1.0)


@unittest.skipUnless(os.path.exists(WALK_ONNX) and os.path.exists(G1_XML),
                     "WBC ONNX policy / MuJoCo XML assets not present in this checkout")
class TestWBCTeacherIntegration(unittest.TestCase):
    """Small, cheap (~0.4 ms/step per docs/teacher.md) integration smokes."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.teacher = WBCTeacher(use_gpu=False)
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBCTeacher unavailable: {e}")

    def test_reset_sets_standing_height(self) -> None:
        self.teacher.reset()
        self.assertAlmostEqual(self.teacher.base_height, RESET_HEIGHT, places=3)

    def test_reset_with_pose(self) -> None:
        self.teacher.reset(pos_xy=(1.5, -2.0), yaw=0.4)
        np.testing.assert_allclose(self.teacher.base_pos[:2], [1.5, -2.0], atol=1e-6)
        self.assertAlmostEqual(self.teacher.base_yaw, 0.4, places=5)

    def test_step_returns_15_joint_targets(self) -> None:
        self.teacher.reset()
        targets = self.teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        self.assertEqual(targets.shape, (NUM_ACTIONS,))

    def test_step_advances_sim_time_by_control_dt(self) -> None:
        self.teacher.reset()
        t0 = self.teacher.sim_time
        self.teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        self.assertAlmostEqual(self.teacher.sim_time - t0, SIM_DT * CONTROL_DECIMATION, places=6)

    def test_zero_cmd_keeps_robot_upright_briefly(self) -> None:
        self.teacher.reset()
        for _ in range(10):
            self.teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        self.assertGreater(self.teacher.base_height, 0.5)

    def test_reset_clears_history_and_action(self) -> None:
        self.teacher.reset()
        for _ in range(5):
            self.teacher.step(vel_cmd=(0.3, 0.0, 0.2))
        self.teacher.reset()
        np.testing.assert_allclose(self.teacher._action, np.zeros(NUM_ACTIONS), atol=1e-9)
        for frame in self.teacher._obs_history:
            np.testing.assert_allclose(frame, 0.0)


if __name__ == "__main__":
    unittest.main()
