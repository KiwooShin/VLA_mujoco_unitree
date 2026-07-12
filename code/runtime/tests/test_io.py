"""Unit tests for code.runtime.io (RolloutResult defaults, keyframe loading,
checkpoint-free model construction)."""

from __future__ import annotations

import os
import unittest

import torch

from code.runtime.io import RolloutResult, load_keyframe, build_model


class TestRolloutResultDefaults(unittest.TestCase):
    def test_required_fields_only(self):
        r = RolloutResult(success=True, failure_tag='success', steps=10,
                          final_dist=0.1, fell=False, upright=True,
                          ms_per_step=5.0, grounding_hz=5.0)
        self.assertEqual(r.goal_source, 'learned')
        self.assertEqual(r.vel_source, 'predicted')
        self.assertFalse(r.residual_action)
        self.assertEqual(r.action_osc_std, 0.0)
        self.assertEqual(r.forward_disp, 0.0)
        self.assertEqual(r.scene_cfg, {})
        self.assertIsNone(r.video_path)
        self.assertEqual(r.stall_break_triggers, 0)
        self.assertEqual(r.avoid_bias_active_frac, 0.0)

    def test_scene_cfg_default_factory_is_independent_per_instance(self):
        a = RolloutResult(success=True, failure_tag='success', steps=1,
                          final_dist=0.0, fell=False, upright=True,
                          ms_per_step=0.0, grounding_hz=0.0)
        b = RolloutResult(success=True, failure_tag='success', steps=1,
                          final_dist=0.0, fell=False, upright=True,
                          ms_per_step=0.0, grounding_hz=0.0)
        a.scene_cfg['x'] = 1
        self.assertEqual(b.scene_cfg, {})


class TestLoadKeyframe(unittest.TestCase):
    def test_use_keyframe_false_returns_none(self):
        self.assertIsNone(load_keyframe(use_keyframe=False))

    def test_missing_file_returns_none(self):
        import code.runtime.io as io_mod
        orig = io_mod.KEYFRAME_PATH
        io_mod.KEYFRAME_PATH = "/nonexistent/path/stand_keyframe.npz"
        try:
            self.assertIsNone(load_keyframe(use_keyframe=True))
        finally:
            io_mod.KEYFRAME_PATH = orig

    def test_real_keyframe_loads_if_present(self):
        from code.runtime.constants import KEYFRAME_PATH
        if not os.path.isfile(KEYFRAME_PATH):
            self.skipTest(f"no keyframe at {KEYFRAME_PATH} in this environment")
        kf = load_keyframe(use_keyframe=True)
        self.assertIsNotNone(kf)
        self.assertIn('qpos_local', kf)
        self.assertIn('qvel_local', kf)
        self.assertIn('target_dof', kf)
        self.assertIn('height', kf)


class TestBuildModelRandomInit(unittest.TestCase):
    """No checkpoint -> random-init GroundedNav; exercises arch A/C wiring
    without needing a real .pt file."""

    def test_random_init_arch_a(self):
        result = build_model(checkpoint_path=None, arch='A', chunk_H=1,
                             device=torch.device('cpu'), goal_source='classical',
                             vel_source='predicted')
        self.assertFalse(result.checkpoint_loaded)
        self.assertEqual(result.arch, 'A')
        self.assertEqual(result.chunk_H, 1)
        self.assertIsNone(result.action_stats)
        self.assertFalse(result.use_phase)
        self.assertFalse(result.grounding_trained)
        self.assertFalse(result.vel_proprio)
        self.assertIsNotNone(result.model)

    def test_random_init_arch_c(self):
        result = build_model(checkpoint_path=None, arch='C', chunk_H=1,
                             device=torch.device('cpu'), goal_source='learned',
                             vel_source='predicted')
        self.assertEqual(result.arch, 'C')
        self.assertFalse(result.checkpoint_loaded)

    def test_nonexistent_checkpoint_path_falls_back_to_random_init(self):
        result = build_model(checkpoint_path='/nonexistent/ckpt.pt', arch='A', chunk_H=1,
                             device=torch.device('cpu'), goal_source='gt',
                             vel_source='predicted')
        self.assertFalse(result.checkpoint_loaded)
        self.assertIsNotNone(result.model)

    def test_teacher_forcing_wired_for_gt_goal_source(self):
        """arch='A' + goal_source='gt' must set teacher_forcing=True on the
        model (gt_goal injection path) — a provable proxy: the model accepts
        a gt_goal tensor without raising when goal is injected."""
        result = build_model(checkpoint_path=None, arch='A', chunk_H=1,
                             device=torch.device('cpu'), goal_source='gt',
                             vel_source='predicted')
        self.assertTrue(result.model.teacher_forcing)

    def test_no_teacher_forcing_for_pure_learned_goal_and_vel(self):
        result = build_model(checkpoint_path=None, arch='A', chunk_H=1,
                             device=torch.device('cpu'), goal_source='learned',
                             vel_source='predicted')
        self.assertFalse(result.model.teacher_forcing)


if __name__ == "__main__":
    unittest.main()
