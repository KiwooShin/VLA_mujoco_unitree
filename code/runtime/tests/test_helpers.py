"""Unit tests for code.runtime.helpers (proprio vector layout, student PD
torques, RGB->tensor conversion, CAM-2 demo-viz labeling).

`_build_proprio`/`_apply_student_pd` only ever index `.qpos`/`.qvel`/`.ctrl`
as plain arrays on the `data` argument, so a lightweight stand-in (no real
MuJoCo/EGL needed) exercises the exact same code paths.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from code.runtime.helpers import _build_proprio, _apply_student_pd, _rgb_to_tensor, _label_active_cam
from code.runtime.constants import PROPRIO_DIM
from code.sim.teacher import NUM_ACTIONS, KPS, KDS


class _FakeMjData:
    """Stand-in for mujoco.MjData exposing only qpos/qvel/ctrl as arrays."""

    def __init__(self, nq: int = 30, nv: int = 30, nu: int = 15):
        self.qpos = np.arange(nq, dtype=np.float64) * 0.01
        self.qvel = np.arange(nv, dtype=np.float64) * 0.02 - 0.1
        self.ctrl = np.zeros(nu, dtype=np.float64)


class TestBuildProprio(unittest.TestCase):
    def test_shape_and_dtype(self):
        data = _FakeMjData()
        prev_action = np.linspace(0.0, 1.0, 15, dtype=np.float32)
        p = _build_proprio(data, prev_action)
        self.assertEqual(p.shape, (PROPRIO_DIM,))
        self.assertEqual(p.dtype, np.float32)

    def test_layout_matches_dataset_spec(self):
        """[0:15] qpos joints, [15:30] qvel joints, [30:34] quat, [34:37] ang
        vel, [37:40] lin vel, [40:55] prev_action — exact slices from
        dataset.md / gen_dataset.py."""
        data = _FakeMjData()
        prev_action = np.full(15, 7.0, dtype=np.float32)
        p = _build_proprio(data, prev_action)
        np.testing.assert_allclose(p[0:15],  data.qpos[7:22],  atol=1e-6)
        np.testing.assert_allclose(p[15:30], data.qvel[6:21],  atol=1e-6)
        np.testing.assert_allclose(p[30:34], data.qpos[3:7],   atol=1e-6)
        np.testing.assert_allclose(p[34:37], data.qvel[3:6],   atol=1e-6)
        np.testing.assert_allclose(p[37:40], data.qvel[0:3],   atol=1e-6)
        np.testing.assert_allclose(p[40:55], prev_action,      atol=1e-6)


class TestApplyStudentPd(unittest.TestCase):
    def test_leg_torque_formula(self):
        data = _FakeMjData()
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float64)
        _apply_student_pd(data, target_dof, nj=NUM_ACTIONS)
        expected = ((target_dof - data.qpos[7:7 + NUM_ACTIONS]) * KPS
                    + (0.0 - data.qvel[6:6 + NUM_ACTIONS]) * KDS)
        np.testing.assert_allclose(data.ctrl[:NUM_ACTIONS], expected, atol=1e-9)

    def test_no_upper_body_torque_when_nj_equals_num_actions(self):
        data = _FakeMjData(nu=NUM_ACTIONS)
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float64)
        _apply_student_pd(data, target_dof, nj=NUM_ACTIONS)
        # ctrl array is exactly NUM_ACTIONS long; no IndexError, no extra writes.
        self.assertEqual(len(data.ctrl), NUM_ACTIONS)

    def test_upper_body_torque_when_nj_exceeds_num_actions(self):
        n_upper = 4
        nj = NUM_ACTIONS + n_upper
        data = _FakeMjData(nq=7 + nj, nv=6 + nj, nu=nj)
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float64)
        _apply_student_pd(data, target_dof, nj=nj)
        expected_arm = ((0.0 - data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
                        + (0.0 - data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5)
        np.testing.assert_allclose(data.ctrl[NUM_ACTIONS:nj], expected_arm, atol=1e-9)


class TestRgbToTensor(unittest.TestCase):
    def test_already_correct_size_no_resize(self):
        from code.runtime.constants import IMG_SIZE
        rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        t = _rgb_to_tensor(rgb, torch.device('cpu'))
        self.assertEqual(tuple(t.shape), (1, 3, IMG_SIZE, IMG_SIZE))
        self.assertAlmostEqual(float(t[0, 0, 0, 0]), 1.0, places=5)

    def test_resizes_when_shape_mismatched(self):
        from code.runtime.constants import IMG_SIZE
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        t = _rgb_to_tensor(rgb, torch.device('cpu'))
        self.assertEqual(tuple(t.shape), (1, 3, IMG_SIZE, IMG_SIZE))

    def test_values_scaled_to_unit_range(self):
        from code.runtime.constants import IMG_SIZE
        rgb = np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)
        t = _rgb_to_tensor(rgb, torch.device('cpu'))
        self.assertTrue(torch.all(t <= 1.0001))
        self.assertTrue(torch.all(t >= 0.0))


class TestLabelActiveCam(unittest.TestCase):
    def test_output_shape_preserved_without_resize(self):
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        out = _label_active_cam(rgb, 'GROUNDING', 2.5)
        self.assertEqual(out.shape, rgb.shape)

    def test_resize_to_changes_output_shape(self):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        out = _label_active_cam(rgb, 'PROXIMITY', 1.1, resize_to=(120, 90))
        self.assertEqual(out.shape, (90, 120, 3))

    def test_does_not_mutate_input(self):
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        rgb_copy = rgb.copy()
        _label_active_cam(rgb, 'GROUNDING', 2.5)
        np.testing.assert_array_equal(rgb, rgb_copy)


if __name__ == "__main__":
    unittest.main()
