"""Unit tests for code.eval.maneuver_types: ManeuverResult, the proprio
builder, the PD control helper, and the video writer -- none need a live
MuJoCo sim (a lightweight numpy stand-in for mujoco.MjData suffices).
"""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from code.eval.maneuver_types import (
    ManeuverResult, FALL_HEIGHT, PROPRIO_K, HEADING_SUCCESS_THR, IMG_SIZE,
    HOLD_STEPS_REQUIRED,
    _build_proprio_maneuver, _apply_student_pd, _write_video,
)
from code.gen_dart_dataset import GaitPhaseTracker
from code.teacher import KPS, KDS, NUM_ACTIONS


class TestConstants(unittest.TestCase):
    def test_fall_height(self):
        self.assertEqual(FALL_HEIGHT, 0.50)

    def test_heading_success_thr_is_25_degrees(self):
        self.assertAlmostEqual(math.degrees(HEADING_SUCCESS_THR), 25.0, places=6)

    def test_proprio_k_and_img_size(self):
        self.assertEqual(PROPRIO_K, 6)
        self.assertEqual(IMG_SIZE, 128)

    def test_hold_steps_required(self):
        self.assertEqual(HOLD_STEPS_REQUIRED, 5)


class TestManeuverResult(unittest.TestCase):
    def _make(self, **overrides) -> ManeuverResult:
        base = dict(
            success=True, failure_tag='success', steps=1400, fell=False,
            upright=True, landmark_passed=True, final_heading_err=0.1,
            final_state=2, ms_per_step=1.5,
        )
        base.update(overrides)
        return ManeuverResult(**base)

    def test_defaults(self):
        r = self._make()
        self.assertEqual(r.scene_cfg, {})
        self.assertIsNone(r.video_path)

    def test_scene_cfg_default_factory_is_independent_per_instance(self):
        a = self._make()
        b = self._make()
        a.scene_cfg['x'] = 1
        self.assertEqual(a.scene_cfg, {'x': 1})
        self.assertEqual(b.scene_cfg, {})

    def test_asdict_json_serializable(self):
        r = self._make(scene_cfg={'turn_direction': 'left'}, video_path='eval/x.mp4')
        d = dataclasses.asdict(r)
        json.dumps(d)
        self.assertEqual(d['scene_cfg']['turn_direction'], 'left')


def _fake_mjdata(nq: int = 30):
    """Lightweight stand-in for mujoco.MjData exposing qpos/qvel/ctrl."""
    return SimpleNamespace(
        qpos=np.zeros(nq, dtype=np.float64),
        qvel=np.zeros(nq, dtype=np.float64),
        ctrl=np.zeros(nq, dtype=np.float64),
    )


class TestBuildProprioManeuver(unittest.TestCase):
    def test_output_shape_is_62(self):
        data = _fake_mjdata()
        prev_action = np.zeros(15, dtype=np.float32)
        tracker = GaitPhaseTracker()
        priv = dict(subgoal_index=1, cos_target=1.0, sin_target=0.0,
                    heading_err=0.0, landmark_passed=False)
        out = _build_proprio_maneuver(data, prev_action, tracker, priv)
        self.assertEqual(out.shape, (62,))
        self.assertEqual(out.dtype, np.float32)

    def test_first_call_phase_is_zero_one(self):
        """GaitPhaseTracker.update() on a fresh tracker always returns (0.0, 1.0)
        regardless of input -- this pins that the [55:57] phase slice reflects it."""
        data = _fake_mjdata()
        prev_action = np.zeros(15, dtype=np.float32)
        tracker = GaitPhaseTracker()
        priv = dict(subgoal_index=0, cos_target=0.0, sin_target=1.0,
                    heading_err=0.0, landmark_passed=False)
        out = _build_proprio_maneuver(data, prev_action, tracker, priv)
        np.testing.assert_allclose(out[55:57], [0.0, 1.0])

    def test_maneuver_feature_slice_matches_priv_dict(self):
        data = _fake_mjdata()
        prev_action = np.zeros(15, dtype=np.float32)
        tracker = GaitPhaseTracker()
        priv = dict(subgoal_index=2, cos_target=0.5, sin_target=-0.5,
                    heading_err=math.pi / 2, landmark_passed=True)
        out = _build_proprio_maneuver(data, prev_action, tracker, priv)
        man = out[57:62]
        self.assertAlmostEqual(float(man[0]), 2.0 / 2.0, places=6)
        self.assertAlmostEqual(float(man[1]), 0.5, places=6)
        self.assertAlmostEqual(float(man[2]), -0.5, places=6)
        self.assertAlmostEqual(float(man[3]), 0.5, places=6)   # (pi/2)/pi
        self.assertAlmostEqual(float(man[4]), 1.0, places=6)

    def test_prev_action_folds_into_base_proprio(self):
        data = _fake_mjdata()
        prev_action = np.arange(15, dtype=np.float32)
        tracker = GaitPhaseTracker()
        priv = dict(subgoal_index=0, cos_target=1.0, sin_target=0.0,
                    heading_err=0.0, landmark_passed=False)
        out = _build_proprio_maneuver(data, prev_action, tracker, priv)
        np.testing.assert_allclose(out[40:55], prev_action)


class TestApplyStudentPd(unittest.TestCase):
    def test_leg_torque_matches_pd_law_at_zero_state(self):
        data = _fake_mjdata()
        target_dof = np.linspace(-0.2, 0.2, NUM_ACTIONS).astype(np.float32)
        _apply_student_pd(data, target_dof, nj=NUM_ACTIONS)
        expected = target_dof * KPS   # qpos/qvel are zero -> KDS term vanishes
        np.testing.assert_allclose(data.ctrl[:NUM_ACTIONS], expected, rtol=1e-5)

    def test_leg_torque_accounts_for_nonzero_qpos_qvel(self):
        data = _fake_mjdata()
        data.qpos[7:7 + NUM_ACTIONS] = 0.05
        data.qvel[6:6 + NUM_ACTIONS] = 0.1
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float32)
        _apply_student_pd(data, target_dof, nj=NUM_ACTIONS)
        expected = (target_dof - 0.05) * KPS + (0.0 - 0.1) * KDS
        np.testing.assert_allclose(data.ctrl[:NUM_ACTIONS], expected, rtol=1e-5)

    def test_arm_torque_applied_when_nj_exceeds_num_actions(self):
        nj = NUM_ACTIONS + 4
        data = _fake_mjdata(nq=7 + nj + 1)
        data.qpos[7 + NUM_ACTIONS:7 + nj] = 0.1
        data.qvel[6 + NUM_ACTIONS:6 + nj] = 0.2
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float32)
        _apply_student_pd(data, target_dof, nj=nj)
        expected_arm = (0.0 - 0.1) * 100.0 + (0.0 - 0.2) * 0.5
        np.testing.assert_allclose(
            data.ctrl[NUM_ACTIONS:nj], np.full(4, expected_arm), rtol=1e-5)

    def test_no_arm_torque_when_nj_equals_num_actions(self):
        data = _fake_mjdata()
        target_dof = np.zeros(NUM_ACTIONS, dtype=np.float32)
        _apply_student_pd(data, target_dof, nj=NUM_ACTIONS)
        # ctrl beyond NUM_ACTIONS untouched (stays zero-initialized)
        np.testing.assert_allclose(data.ctrl[NUM_ACTIONS:], 0.0)


class TestWriteVideo(unittest.TestCase):
    def test_writes_ego_only_when_no_third_person(self):
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / 'clip.mp4')
            _write_video(frames, [], out_path, fps=10)
            self.assertTrue(Path(out_path).exists())
            self.assertGreater(Path(out_path).stat().st_size, 0)

    def test_writes_side_by_side_when_tp_frames_present(self):
        frames_ego = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        frames_tp = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / 'clip_sbs.mp4')
            _write_video(frames_ego, frames_tp, out_path, fps=10)
            self.assertTrue(Path(out_path).exists())


if __name__ == '__main__':
    unittest.main()
