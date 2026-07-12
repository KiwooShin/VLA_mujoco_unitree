"""Unit tests for code.runtime.gt_goal._compute_gt_goal (privileged GT goal
probe) — analytic cases vs. hand-derived (dist, cos, sin) values."""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.runtime.gt_goal import _compute_gt_goal


class _FakeMjData:
    """Stand-in for mujoco.MjData: qpos[0:2]=xy, qpos[3:7]=quat [w,x,y,z]."""

    def __init__(self, x: float, y: float, yaw: float):
        self.qpos = np.zeros(7, dtype=np.float64)
        self.qpos[0] = x
        self.qpos[1] = y
        self.qpos[3] = math.cos(yaw / 2)  # w
        self.qpos[4] = 0.0
        self.qpos[5] = 0.0
        self.qpos[6] = math.sin(yaw / 2)  # z


class TestComputeGtGoal(unittest.TestCase):
    def test_target_directly_ahead(self):
        data = _FakeMjData(0.0, 0.0, yaw=0.0)
        goal = _compute_gt_goal(data, np.array([3.0, 0.0]))
        dist, cos_th, sin_th = goal
        self.assertAlmostEqual(dist, 3.0, places=5)
        self.assertAlmostEqual(cos_th, 1.0, places=5)
        self.assertAlmostEqual(sin_th, 0.0, places=5)

    def test_target_directly_behind(self):
        data = _FakeMjData(0.0, 0.0, yaw=0.0)
        goal = _compute_gt_goal(data, np.array([-2.0, 0.0]))
        dist, cos_th, sin_th = goal
        self.assertAlmostEqual(dist, 2.0, places=5)
        self.assertAlmostEqual(cos_th, -1.0, places=5)
        self.assertAlmostEqual(abs(sin_th), 0.0, places=4)

    def test_target_to_the_left(self):
        """Target at world +y, robot facing +x (yaw=0): positive yaw_err
        (left) per the docstring convention."""
        data = _FakeMjData(0.0, 0.0, yaw=0.0)
        goal = _compute_gt_goal(data, np.array([0.0, 2.0]))
        dist, cos_th, sin_th = goal
        self.assertAlmostEqual(dist, 2.0, places=5)
        self.assertAlmostEqual(cos_th, 0.0, places=5)
        self.assertGreater(sin_th, 0.0)   # left = positive per module docstring

    def test_target_to_the_right(self):
        data = _FakeMjData(0.0, 0.0, yaw=0.0)
        goal = _compute_gt_goal(data, np.array([0.0, -2.0]))
        dist, cos_th, sin_th = goal
        self.assertAlmostEqual(dist, 2.0, places=5)
        self.assertLess(sin_th, 0.0)

    def test_robot_yaw_rotation_is_undone(self):
        """A target directly ahead of a robot yawed 90 deg (facing +y) is at
        world (0, 3): should report the SAME egocentric (dist, 1, 0) as the
        yaw=0 straight-ahead case."""
        data = _FakeMjData(0.0, 0.0, yaw=math.pi / 2)
        goal = _compute_gt_goal(data, np.array([0.0, 3.0]))
        dist, cos_th, sin_th = goal
        self.assertAlmostEqual(dist, 3.0, places=5)
        self.assertAlmostEqual(cos_th, 1.0, places=4)
        self.assertAlmostEqual(sin_th, 0.0, places=4)

    def test_translation_invariance(self):
        """Only the relative offset matters, not absolute world position."""
        data_a = _FakeMjData(5.0, 5.0, yaw=0.3)
        data_b = _FakeMjData(0.0, 0.0, yaw=0.3)
        goal_a = _compute_gt_goal(data_a, np.array([5.0 + 1.0, 5.0 + 2.0]))
        goal_b = _compute_gt_goal(data_b, np.array([1.0, 2.0]))
        np.testing.assert_allclose(goal_a, goal_b, atol=1e-5)

    def test_zero_distance(self):
        data = _FakeMjData(1.0, 1.0, yaw=0.7)
        goal = _compute_gt_goal(data, np.array([1.0, 1.0]))
        self.assertAlmostEqual(goal[0], 0.0, places=6)

    def test_return_dtype_and_shape(self):
        data = _FakeMjData(0.0, 0.0, yaw=0.0)
        goal = _compute_gt_goal(data, np.array([1.0, 1.0]))
        self.assertEqual(goal.shape, (3,))
        self.assertEqual(goal.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
